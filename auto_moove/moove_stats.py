#!/usr/bin/env python3
"""Bootstrap confidence intervals for the auto_moove pipeline.

Ported from auto_moove_old/moove_common.py. That module held the statistics
half of the pipeline (bootstrap_ci, compute_ci, ...) while auto_moove/
inherited only the generation and judging half, which left the canonical
pipeline printing point estimates with no interval on them.

Resampling unit
---------------
auto_moove.py deduplicates prompts by pair_key in load_prompts(), so each row
of its results frame is already one unique pair. A plain 1-D bootstrap over
those rows is therefore the correct resampling unit here.

This is NOT true of the judged_*.jsonl frames consumed by auto_moove_check.py
or auto_moove_vote_rules.py, where several human reviews share a pair_key and
the bootstrap must resample pair keys rather than rows -- see the cluster
helper below and the existing percentile bootstrap in auto_moove_vote_rules.py.

scipy is optional. With it, compute_ci uses BCa (falling back to scipy's
percentile method). Without it, it falls back to a numpy percentile bootstrap,
which is wider and less accurate in the tails but is still a real interval --
the container is not guaranteed to ship scipy, and a run launched with
--requirements none cannot install it, because PYTHONNOUSERSITE=1 hides
anything pip puts in ~/.local. Only a sample too small to resample (n < 2)
yields (None, None), which fmt_ci renders as a bare point estimate.
"""

import numpy as np

try:
    from scipy.stats import bootstrap as _scipy_bootstrap
    HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    _scipy_bootstrap = None
    HAVE_SCIPY = False


def bootstrap_ci(data_tuple, stat_fn, *, paired=False, vectorized=True,
                 n_resamples=1000):
    """95% CI. Try BCa, fall back to percentile, then to (None, None)."""
    if not HAVE_SCIPY:
        return None, None
    for method in ("BCa", "percentile"):
        try:
            res = _scipy_bootstrap(
                data_tuple, stat_fn,
                paired=paired, vectorized=vectorized,
                n_resamples=n_resamples, confidence_level=0.95,
                method=method,
            )
            return res.confidence_interval.low, res.confidence_interval.high
        except Exception:
            continue
    return None, None


def compute_ci(data):
    """Point estimate (mean) and 95% bootstrap CI for a 1-D sample.

    Returns (point_estimate, (lo, hi)). (lo, hi) is (None, None) when the CI
    could not be computed -- too few points, or scipy unavailable.
    """
    data = [d for d in np.asarray(data, dtype=float).ravel() if not np.isnan(d)]
    if len(data) < 2:
        val = float(data[0]) if data else 0.0
        return val, (None, None)

    arr = np.asarray(data, dtype=float)
    point_est = float(np.mean(arr))

    lo, hi = bootstrap_ci((arr,), np.mean)
    if lo is not None:
        return point_est, (lo, hi)

    # scipy missing (or BCa and percentile both failed): fall back to a plain
    # numpy percentile bootstrap. Wider and less accurate in the tails than
    # BCa, but a real interval beats no interval -- the container is not
    # guaranteed to ship scipy, and runs launched with --requirements none
    # cannot install it (PYTHONNOUSERSITE hides pip --user).
    return point_est, _percentile_bootstrap(arr)


def _percentile_bootstrap(arr, n_resamples=1000, rng=None):
    """95% percentile bootstrap of the mean, numpy only."""
    if arr.size < 2:
        return (None, None)
    rng = rng or np.random.default_rng(2001)
    draws = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def compute_ci_clustered(values, cluster_keys, n_resamples=1000, rng=None):
    """Mean and 95% percentile CI, resampling whole clusters with replacement.

    For frames where rows are not independent -- several human reviews sharing
    one pair_key. Resampling rows there understates the interval.
    """
    values = np.asarray(values, dtype=float)
    keys = np.asarray(cluster_keys)
    ok = ~np.isnan(values)
    values, keys = values[ok], keys[ok]
    if values.size < 2:
        return (float(values[0]) if values.size else 0.0), (None, None)

    point = float(values.mean())
    uniq = np.unique(keys)
    if uniq.size < 2:
        return point, (None, None)

    rng = rng or np.random.default_rng(2001)
    by_key = {k: values[keys == k] for k in uniq}
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        draws[i] = np.concatenate([by_key[k] for k in pick]).mean()
    return point, (float(np.percentile(draws, 2.5)),
                   float(np.percentile(draws, 97.5)))


def fmt_ci(point, ci, *, pct=False, signed=False, digits=2):
    """Render 'x.xx [lo, hi]', or the bare point estimate when there is no CI."""
    scale = 100.0 if pct else 1.0
    unit = "%" if pct else ""
    sign = "+" if signed else ""
    body = f"{point * scale:{sign}.{digits}f}{unit}"
    lo, hi = ci
    if lo is None or hi is None:
        return body
    return f"{body} [{lo * scale:{sign}.{digits}f}, {hi * scale:{sign}.{digits}f}]"
