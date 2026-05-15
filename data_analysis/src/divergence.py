"""
Distribution divergence metrics for source<->synthetic comparison.

Three metrics:
  - Jensen-Shannon divergence (base-2, range [0, 1]) for categorical fields
  - Wasserstein-1 distance for ordinal fields (difficulty: 1-5 scale)
  - Total variation distance (range [0, 1]) as a simpler summary

Bootstrap 95% CIs are computed by resampling both arms independently.
At n=2000 per arm with 1000 bootstrap iterations this is fast (~1s per field).

Design notes
------------
- All metrics take two iterables of values. The values are already cleaned by
  the metadata extractor (enum coercion), so no validation happens here.
- "unspecified" is kept as a regular category; it's a meaningful response
  ("the model couldn't tell from context") and its proportion is diagnostic.
- Bootstrap uses a fixed RNG seeded per (field, metric) so CIs are reproducible.
- Geography is open-ended — we bin to continents before computing divergence,
  but the raw values are also summarized for reporting.
"""

import hashlib
import math
from collections import Counter
from typing import Iterable, List, Optional, Tuple, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Proportions and divergence math
# ---------------------------------------------------------------------------

def _proportions(values, categories):
    """Return an array of proportions over `categories`, in the same order.

    Values not in `categories` are ignored (should be rare given the extractor
    enforces enums, but possible for geography or any free-text field)."""
    counts = Counter(values)
    total = sum(counts.get(c, 0) for c in categories)
    if total == 0:
        return np.zeros(len(categories))
    return np.array([counts.get(c, 0) / total for c in categories])


def jensen_shannon(p, q, base=2):
    """Jensen-Shannon divergence between two probability vectors."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl(a, b):
        # Element-wise KL with 0 * log(0) = 0; a_i = 0 contributes 0.
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * (np.log(a[mask]) - np.log(b[mask]))))

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    if base == 2:
        jsd = jsd / math.log(2)
    # Clamp to [0, 1] — floating error can push this tiny amounts negative
    return max(0.0, min(1.0, jsd))


def total_variation(p, q):
    """Total variation distance between two probability vectors."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return 0.5 * float(np.sum(np.abs(p - q)))


def wasserstein_1(values_a, values_b):
    """Wasserstein-1 distance between two empirical ordinal distributions.

    For 1-D ordinal data this is just the L1 distance between the two ECDFs.
    Uses numpy only (no scipy dependency), so works in minimal containers.
    """
    a = np.sort(np.asarray(values_a, dtype=np.float64))
    b = np.sort(np.asarray(values_b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    # Stack all points, compute ECDFs at each point
    all_points = np.sort(np.concatenate([a, b]))
    # ECDF values at each point
    ecdf_a = np.searchsorted(a, all_points, side="right") / len(a)
    ecdf_b = np.searchsorted(b, all_points, side="right") / len(b)
    # Trapezoidal-style integration of |ECDF_a - ECDF_b| over the real line
    diffs = np.abs(ecdf_a[:-1] - ecdf_b[:-1])
    widths = np.diff(all_points)
    return float(np.sum(diffs * widths))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _seed_for(*parts):
    """Deterministic int seed from a tuple of strings."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"\0")
    return int.from_bytes(h.digest()[:4], "big")


def bootstrap_ci(metric_fn, values_a, values_b, n_iter=1000, alpha=0.05, seed_parts=()):
    """Generic bootstrap CI. metric_fn(values_a_sample, values_b_sample) -> float."""
    rng = np.random.default_rng(_seed_for(*seed_parts))
    n_a = len(values_a)
    n_b = len(values_b)
    if n_a == 0 or n_b == 0:
        return float("nan"), float("nan"), float("nan")
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    stats = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        stats[i] = metric_fn(values_a[idx_a], values_b[idx_b])
    point = metric_fn(values_a, values_b)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return float(point), lo, hi


# ---------------------------------------------------------------------------
# Categorical divergence (JSD + TV)
# ---------------------------------------------------------------------------

def categorical_divergence(values_a, values_b, categories, n_iter=1000, seed_parts=()):
    """Return (jsd_point, jsd_lo, jsd_hi, tv_point, tv_lo, tv_hi, props_a, props_b)."""
    def jsd_fn(a, b):
        return jensen_shannon(_proportions(a, categories), _proportions(b, categories))

    def tv_fn(a, b):
        return total_variation(_proportions(a, categories), _proportions(b, categories))

    jsd_p, jsd_lo, jsd_hi = bootstrap_ci(
        jsd_fn, values_a, values_b, n_iter=n_iter,
        seed_parts=tuple(seed_parts) + ("jsd",),
    )
    tv_p, tv_lo, tv_hi = bootstrap_ci(
        tv_fn, values_a, values_b, n_iter=n_iter,
        seed_parts=tuple(seed_parts) + ("tv",),
    )
    props_a = _proportions(values_a, categories)
    props_b = _proportions(values_b, categories)
    return jsd_p, jsd_lo, jsd_hi, tv_p, tv_lo, tv_hi, props_a, props_b


# ---------------------------------------------------------------------------
# Ordinal divergence (Wasserstein)
# ---------------------------------------------------------------------------

def ordinal_divergence(values_a, values_b, n_iter=1000, seed_parts=()):
    """Return (w1_point, w1_lo, w1_hi, mean_a, mean_b, median_a, median_b)."""
    values_a = np.asarray([v for v in values_a if v is not None], dtype=np.float64)
    values_b = np.asarray([v for v in values_b if v is not None], dtype=np.float64)
    if len(values_a) == 0 or len(values_b) == 0:
        return (float("nan"),) * 7

    w1_p, w1_lo, w1_hi = bootstrap_ci(
        wasserstein_1, values_a, values_b, n_iter=n_iter,
        seed_parts=tuple(seed_parts) + ("w1",),
    )
    return (
        w1_p, w1_lo, w1_hi,
        float(np.mean(values_a)), float(np.mean(values_b)),
        float(np.median(values_a)), float(np.median(values_b)),
    )


# ---------------------------------------------------------------------------
# Geography binning
# ---------------------------------------------------------------------------

# Minimal country->continent mapping covering the countries we see in our
# data. Anything not listed falls through to the normalization rules below
# and eventually to "other" if nothing matches.
_CONTINENT_MAP = {
    # Americas
    "united states": "americas", "us": "americas", "usa": "americas",
    "canada": "americas", "mexico": "americas", "brazil": "americas",
    "argentina": "americas", "chile": "americas", "peru": "americas",
    "colombia": "americas", "venezuela": "americas", "bolivia": "americas",
    "ecuador": "americas", "cuba": "americas", "haiti": "americas",
    "dominican republic": "americas", "jamaica": "americas",
    "north america": "americas", "south america": "americas",
    "latin america": "americas", "caribbean": "americas",
    # Europe
    "united kingdom": "europe", "uk": "europe", "england": "europe",
    "scotland": "europe", "wales": "europe", "ireland": "europe",
    "france": "europe", "germany": "europe", "italy": "europe",
    "spain": "europe", "portugal": "europe", "netherlands": "europe",
    "belgium": "europe", "switzerland": "europe", "austria": "europe",
    "sweden": "europe", "norway": "europe", "denmark": "europe",
    "finland": "europe", "poland": "europe", "greece": "europe",
    "russia": "europe", "ukraine": "europe", "romania": "europe",
    "czech republic": "europe", "hungary": "europe",
    "europe": "europe",
    # Africa
    "nigeria": "africa", "egypt": "africa", "south africa": "africa",
    "kenya": "africa", "ethiopia": "africa", "ghana": "africa",
    "uganda": "africa", "tanzania": "africa", "morocco": "africa",
    "algeria": "africa", "sudan": "africa", "zimbabwe": "africa",
    "senegal": "africa", "rwanda": "africa", "cameroon": "africa",
    "malawi": "africa", "mozambique": "africa", "zambia": "africa",
    "angola": "africa", "ivory coast": "africa", "cote d'ivoire": "africa",
    "democratic republic of congo": "africa", "drc": "africa",
    "africa": "africa", "sub-saharan africa": "africa",
    # Asia
    "china": "asia", "india": "asia", "japan": "asia", "south korea": "asia",
    "north korea": "asia", "thailand": "asia", "vietnam": "asia",
    "indonesia": "asia", "philippines": "asia", "malaysia": "asia",
    "singapore": "asia", "pakistan": "asia", "bangladesh": "asia",
    "sri lanka": "asia", "nepal": "asia", "myanmar": "asia",
    "cambodia": "asia", "laos": "asia", "mongolia": "asia",
    "turkey": "asia", "iran": "asia", "iraq": "asia", "israel": "asia",
    "saudi arabia": "asia", "united arab emirates": "asia", "uae": "asia",
    "jordan": "asia", "lebanon": "asia", "syria": "asia",
    "asia": "asia", "middle east": "asia", "southeast asia": "asia",
    "south asia": "asia", "east asia": "asia",
    # Oceania
    "australia": "oceania", "new zealand": "oceania", "fiji": "oceania",
    "papua new guinea": "oceania", "oceania": "oceania",
    # Global / unspecified
    "global": "global", "worldwide": "global", "international": "global",
}


def bin_geography(raw_value):
    """Normalize a free-text geography tag to one of:
    americas, europe, africa, asia, oceania, global, unspecified, other."""
    if raw_value is None:
        return "unspecified"
    v = str(raw_value).strip().lower()
    if not v or v in ("unspecified", "n/a", "none", "null", "not specified"):
        return "unspecified"
    # Exact match
    if v in _CONTINENT_MAP:
        return _CONTINENT_MAP[v]
    # Substring match (handles "rural Brazil", "northeastern Thailand")
    for country, continent in _CONTINENT_MAP.items():
        if country in v:
            return continent
    return "other"


GEOGRAPHY_BINS = [
    "americas", "europe", "africa", "asia", "oceania",
    "global", "other", "unspecified",
]


# ---------------------------------------------------------------------------
# Task-format shift comparison
# ---------------------------------------------------------------------------

def task_format_stats(records):
    """Return summary stats useful for the 'task-format shift' comparison.

    Takes an iterable of Record-dict-likes (as written by stage 1)."""
    n = 0
    mcq = 0
    paired = 0
    doc = 0
    think = 0
    q_lens = []
    a_lens = []
    full_lens = []

    for r in records:
        n += 1
        if r.get("is_mcq"):
            mcq += 1
        if r.get("is_paired"):
            paired += 1
        if r.get("is_document"):
            doc += 1
        if r.get("has_think_block"):
            think += 1
        q = r.get("question") or ""
        a = r.get("answer") or ""
        ft = r.get("full_text") or ""
        q_lens.append(len(q.split()))
        a_lens.append(len(a.split()))
        full_lens.append(len(ft.split()))

    def _pct(num):
        return 100.0 * num / n if n else 0.0

    def _median(xs):
        return float(np.median(xs)) if xs else 0.0

    return {
        "n": n,
        "pct_mcq": _pct(mcq),
        "pct_paired": _pct(paired),
        "pct_document": _pct(doc),
        "pct_has_think_block": _pct(think),
        "median_q_tokens": _median(q_lens),
        "median_a_tokens": _median(a_lens),
        "median_full_tokens": _median(full_lens),
    }


# ---------------------------------------------------------------------------
# Schema for reporting
# ---------------------------------------------------------------------------

# Allowed categories per field, for divergence math. Must match
# metadata_extract.py's allowed values.
FIELD_CATEGORIES = {
    "specialty": [
        "general_medicine", "emergency_medicine", "critical_care",
        "infectious_disease", "cardiology", "neurology", "oncology",
        "endocrinology", "gastroenterology", "nephrology", "pulmonology",
        "rheumatology", "hematology", "dermatology", "psychiatry",
        "obstetrics", "gynecology", "pediatrics", "geriatrics",
        "surgery_general", "orthopedics", "urology", "ophthalmology",
        "ent", "radiology", "pathology", "anesthesiology",
        "public_health", "family_medicine", "unspecified",
    ],
    "level_of_care": [
        "community", "prehospital", "primary", "emergency",
        "secondary", "tertiary", "unspecified",
    ],
    "severity": [
        "routine", "urgent", "emergent", "life_threatening", "unspecified",
    ],
    "question_type": [
        "triage", "diagnosis", "management", "follow_up", "prevention",
        "screening", "interpretation", "pharmacology", "ethics", "other",
    ],
    "age_group": [
        "neonate", "infant", "child", "adolescent", "adult",
        "elderly", "mixed", "unspecified",
    ],
    "geography": GEOGRAPHY_BINS,  # binned to continents
}

# Fields present in each record type
QA_FIELDS = ["specialty", "level_of_care", "severity", "question_type",
             "geography", "age_group"]
DOC_FIELDS = ["specialty", "level_of_care", "severity", "geography"]
ORDINAL_FIELDS = ["difficulty"]  # only for QA records
SHARED_FIELDS_FOR_GUIDELINES = DOC_FIELDS  # intersection of source (doc) and synthetic (QA)
