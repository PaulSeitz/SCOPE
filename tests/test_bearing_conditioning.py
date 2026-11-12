"""Bearing-conditioning (action-mix TV across lateral buckets)."""

import numpy as np
import pytest

from concept_based_learning.paper_eval.bearing_conditioning import (
    mean_pairwise_tv,
    signed_lat_bucket,
    total_variation,
)


def test_total_variation_identical_is_zero():
    p = np.array([0.5, 0.5])
    assert total_variation(p, p) == 0.0


def test_total_variation_disjoint_is_one():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert total_variation(p, q) == pytest.approx(1.0)


def test_mean_pairwise_tv_invariant_habit():
    buckets = {
        "a": {0: 50, 1: 50},
        "b": {0: 50, 1: 50},
        "c": {0: 25, 1: 25},
    }
    out = mean_pairwise_tv(buckets, n_actions=2, min_count=1)
    assert out["bearing_tv_mean"] == pytest.approx(0.0)
    assert out["n_buckets_used"] == 3


def test_mean_pairwise_tv_conditioned():
    buckets = {
        "left": {0: 100},
        "right": {1: 100},
    }
    out = mean_pairwise_tv(buckets, n_actions=2, min_count=1)
    assert out["bearing_tv_mean"] == pytest.approx(1.0)


def test_signed_lat_bucket_edges():
    assert signed_lat_bucket(-25.0) == "lat<-20"
    assert signed_lat_bucket(-10.0) == "lat-20..-8"
    assert signed_lat_bucket(0.0) == "|lat|<8"
    assert signed_lat_bucket(15.0) == "lat+8..+20"
    assert signed_lat_bucket(30.0) == "lat>+20"
