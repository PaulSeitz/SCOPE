"""Bearing-conditioning score: does the policy's action mix depend on lateral offset?

Mean pairwise total variation (TV) between action histograms across lateral
buckets. 0 = identical distributions (bearing-invariant habit); ~1 = fully
disjoint. Useful for distinguishing transferred tracking from a Seaquest motor
prior that happens to look like tracking when the ball drifts the right way.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np

DEFAULT_N_ACTIONS = 18


def _to_prob_vec(
    counts: Mapping[Any, int | float],
    *,
    n_actions: int,
    name_to_idx: Mapping[str, int] | None = None,
) -> np.ndarray:
    vec = np.zeros(n_actions, dtype=np.float64)
    for key, val in counts.items():
        if isinstance(key, str):
            if name_to_idx is None or key not in name_to_idx:
                continue
            idx = int(name_to_idx[key])
        else:
            idx = int(key)
        if 0 <= idx < n_actions:
            vec[idx] += float(val)
    total = float(vec.sum())
    if total <= 0.0:
        return np.full(n_actions, 1.0 / n_actions, dtype=np.float64)
    return vec / total


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """TV distance in [0, 1] between two discrete distributions."""
    return float(0.5 * np.abs(p - q).sum())


def mean_pairwise_tv(
    bucket_counts: Mapping[str, Mapping[Any, int | float]],
    *,
    n_actions: int = DEFAULT_N_ACTIONS,
    name_to_idx: Mapping[str, int] | None = None,
    min_count: int = 1,
) -> dict[str, float | int | None]:
    """Mean pairwise TV over buckets with at least ``min_count`` samples.

    Returns keys:
      - bearing_tv_mean
      - bearing_tv_max
      - n_bucket_pairs
      - n_buckets_used
    """
    probs: list[np.ndarray] = []
    for _name, counts in bucket_counts.items():
        n = int(sum(int(v) for v in counts.values()))
        if n < min_count:
            continue
        probs.append(
            _to_prob_vec(counts, n_actions=n_actions, name_to_idx=name_to_idx)
        )
    if len(probs) < 2:
        return {
            "bearing_tv_mean": None,
            "bearing_tv_max": None,
            "n_bucket_pairs": 0,
            "n_buckets_used": len(probs),
        }
    tvs = [total_variation(a, b) for a, b in combinations(probs, 2)]
    return {
        "bearing_tv_mean": float(np.mean(tvs)),
        "bearing_tv_max": float(np.max(tvs)),
        "n_bucket_pairs": len(tvs),
        "n_buckets_used": len(probs),
    }


def signed_lat_bucket(lat: float) -> str:
    """Signed lateral bucket for direction-sensitive conditioning."""
    if lat < -20.0:
        return "lat<-20"
    if lat < -8.0:
        return "lat-20..-8"
    if lat <= 8.0:
        return "|lat|<8"
    if lat <= 20.0:
        return "lat+8..+20"
    return "lat>+20"


SIGNED_LAT_BUCKETS = (
    "lat<-20",
    "lat-20..-8",
    "|lat|<8",
    "lat+8..+20",
    "lat>+20",
)

ABS_LAT_BUCKETS = ("|lat|<8", "|lat|8-20", "|lat|>20")
