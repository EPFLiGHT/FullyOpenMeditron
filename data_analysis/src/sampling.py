"""
Deterministic stratified sampling for (pair, split) groups.

Design goals
------------
- Reproducible: same --seed + same input file = same sample, always.
- Stratum-aware: some groups benefit from stratified sampling (MOOVE source
  by vote, Guidelines source by guideline_source). Others are random.
- Streaming: never materialize the full iterator. MOOVE source is ~25k records
  so it fits, but Meditron source is 216k — we reservoir-sample instead.
- Pluggable strata: the caller provides a key function; this module does the math.
"""

import hashlib
import random
from typing import Callable, Iterator, List, Optional


def _stable_key(seed, record_id):
    """Hash seed + id to a uniform [0, 1) float. Independent of Python's
    PRNG state, so order of iteration doesn't matter."""
    h = hashlib.sha256()
    h.update(str(seed).encode())
    h.update(b"\0")
    h.update(str(record_id).encode())
    # Use first 8 bytes as uint64, normalize to [0, 1)
    return int.from_bytes(h.digest()[:8], "big") / (1 << 64)


def reservoir_sample(records, n, seed=0):
    """Random sample of n records from a streaming iterator.

    Uses algorithm R with hash-based keys for determinism: each record gets
    a fixed key derived from (seed, record.id), and we keep the n records
    with the smallest keys. Result is the same regardless of iteration order.
    """
    if n < 0:
        # Full run: yield everything, no sampling
        for r in records:
            yield r
        return

    if n == 0:
        return

    # Keep a heap of (key, counter, record) — counter is a tiebreak for
    # when two keys are identical (shouldn't happen but be safe).
    import heapq
    heap = []
    counter = 0

    for r in records:
        key = _stable_key(seed, r.id)
        # Negate key so heapq (min-heap) keeps the n SMALLEST keys at top
        entry = (-key, counter, r)
        counter += 1
        if len(heap) < n:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:  # our -key is larger = actual key is smaller
            heapq.heapreplace(heap, entry)

    # Yield in whatever order — downstream doesn't care
    for _, _, r in heap:
        yield r


def stratified_sample(records, n, stratum_key, seed=0, min_per_stratum=1):
    """Stratified sample: n total records, allocated proportional to stratum size.

    Parameters
    ----------
    records : iterable of Record
    n : int
        Target sample size. Pass -1 for full run (yields everything).
    stratum_key : callable(Record) -> str or None
        Returns the stratum label. None means "ungrouped / unknown" — those
        records form their own stratum.
    seed : int
    min_per_stratum : int
        Guarantee at least this many records from each non-empty stratum
        (subject to the stratum actually having that many records).

    Strategy
    --------
    1. Walk records once, group by stratum_key.
    2. Allocate n proportional to stratum size, floor + correct rounding.
    3. reservoir_sample within each stratum with a stratum-specific seed.
    """
    if n < 0:
        for r in records:
            yield r
        return

    # Collect by stratum. This materializes the iterator — fine for our sizes
    # (max ~216k Meditron source records, each a Python object ~1KB = 200MB).
    by_stratum = {}
    for r in records:
        k = stratum_key(r)
        if k is None:
            k = "__unknown__"
        by_stratum.setdefault(k, []).append(r)

    if not by_stratum:
        return

    total_records = sum(len(v) for v in by_stratum.values())
    if total_records <= n:
        # Less data than budget — yield everything
        for strat_records in by_stratum.values():
            for r in strat_records:
                yield r
        return

    # Proportional allocation with minimum per-stratum floor
    strata = sorted(by_stratum.keys())  # deterministic order
    sizes = {k: len(by_stratum[k]) for k in strata}

    # First pass: give each stratum min(min_per_stratum, stratum_size)
    allocation = {k: min(min_per_stratum, sizes[k]) for k in strata}
    remaining = n - sum(allocation.values())
    if remaining < 0:
        # More strata than budget — just take 1 from each up to n
        trimmed = {}
        for i, k in enumerate(strata):
            if i >= n:
                break
            trimmed[k] = 1
        allocation = trimmed
        remaining = 0

    # Second pass: distribute remaining proportionally to (stratum_size - already_allocated)
    if remaining > 0:
        headroom = {k: sizes[k] - allocation[k] for k in strata}
        total_headroom = sum(headroom.values())
        if total_headroom > 0:
            # Floor allocation
            fractional = {}
            for k in strata:
                if total_headroom == 0:
                    continue
                share = remaining * headroom[k] / total_headroom
                allocation[k] += int(share)
                fractional[k] = share - int(share)

            # Distribute rounding leftover to strata with largest fractional parts
            leftover = n - sum(allocation.values())
            for k, _ in sorted(fractional.items(), key=lambda kv: -kv[1])[:leftover]:
                if allocation[k] < sizes[k]:
                    allocation[k] += 1

    # Final pass: sample within each stratum
    for k in strata:
        take = allocation[k]
        if take <= 0:
            continue
        strat_seed = "{}::{}".format(seed, k)
        for r in reservoir_sample(iter(by_stratum[k]), take, seed=strat_seed):
            yield r


# ---------------------------------------------------------------------------
# Per-group stratum key functions
# ---------------------------------------------------------------------------

def stratum_for_moove_source(r):
    """MOOVE source: stratify by vote (1, 2, 12, unknown)."""
    return r.vote or "unknown"


def stratum_for_meditron_synthetic(r):
    """Meditron synthetic: stratify by MCQ-ness (MCQ vs open-ended)."""
    return "mcq" if r.is_mcq else "open"


def stratum_for_guidelines_source(r):
    """Guidelines source: stratify by which source file (aafp/cps/drugs/etc)."""
    return r.guideline_source or "unknown"


def no_stratum(r):
    """Sentinel for groups where we just want random sampling."""
    return "all"


# ---------------------------------------------------------------------------
# Dispatch: pick the right sampler for a (pair, split) group
# ---------------------------------------------------------------------------

def sample_group(records, pair, split, n, seed=0):
    """Top-level entry: sample n records from a group, using stratification
    appropriate for that (pair, split)."""
    if (pair, split) == ("moove", "source"):
        return stratified_sample(records, n, stratum_for_moove_source, seed=seed)
    if (pair, split) == ("meditron", "synthetic"):
        return stratified_sample(records, n, stratum_for_meditron_synthetic, seed=seed)
    if (pair, split) == ("guidelines", "source"):
        return stratified_sample(records, n, stratum_for_guidelines_source, seed=seed)
    # Default: simple random reservoir sample
    return reservoir_sample(records, n, seed=seed)
