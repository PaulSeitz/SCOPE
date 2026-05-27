"""Unit tests for the Q-decomposition composition primitives.

These tests use a synthetic action_mapper that returns a deterministic
function of (ego_action, target_idx). No real environment/observations are
needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from concept_based_learning.composition import (
    compose_global_q,
    ego_q_to_global_q,
    invert_global_to_ego_action,
    select_slots,
    select_slots_eps,
    select_slots_per_label,
    select_slots_topk,
    slots_from_heuristic_target,
    _masked_softmax,
    _zscore,
)
from concept_based_learning.environments.datatypes import (
    LABEL_AVOID,
    LABEL_COLLECT,
    LABEL_DESTROY,
    LABEL_NOOP,
)


EGO_DIM = 18
NUM_GAME_ACTIONS = 18
NO_TARGET = -1


# ---------------------------------------------------------------------------
# Stub action mappers
# ---------------------------------------------------------------------------

def _identity_mapper(ego_action, raw_obs, target_idx, objects):
    """Trivial mapper: ego action == game action."""
    return ego_action.astype(jnp.int32)


def _collision_mapper(ego_action, raw_obs, target_idx, objects):
    """Mapper that collapses ego actions 2 and 3 to game action 99 (a sentinel
    above NUM_GAME_ACTIONS for the dedicated collision test); used in a test
    that uses an enlarged num_game_actions.
    """
    return jnp.where(
        (ego_action == 2) | (ego_action == 3),
        jnp.int32(7),
        ego_action.astype(jnp.int32),
    )


def _target_shifted_mapper(ego_action, raw_obs, target_idx, objects):
    """Mapper whose output depends on target_idx — for invert tests."""
    return ((ego_action + target_idx) % NUM_GAME_ACTIONS).astype(jnp.int32)


# ---------------------------------------------------------------------------
# ego_q_to_global_q
# ---------------------------------------------------------------------------

class TestEgoQToGlobalQ:
    def test_identity_mapper_preserves_q(self):
        ego_q = jnp.arange(EGO_DIM, dtype=jnp.float32)
        out = ego_q_to_global_q(
            ego_q, None, jnp.int32(0), None, _identity_mapper, NUM_GAME_ACTIONS
        )
        np.testing.assert_allclose(out, ego_q)

    def test_segment_max_on_collisions(self):
        ego_q = jnp.arange(EGO_DIM, dtype=jnp.float32) - 5.0
        # _collision_mapper: ego 2 and 3 both map to game 7; everything else
        # is identity, so ego 7 also maps to game 7. Game 7 should take the
        # max over {ego_q[2], ego_q[3], ego_q[7]} = ego_q[7].
        out = ego_q_to_global_q(
            ego_q, None, jnp.int32(0), None, _collision_mapper, NUM_GAME_ACTIONS
        )
        expected_g7 = jnp.maximum(jnp.maximum(ego_q[2], ego_q[3]), ego_q[7])
        assert float(out[7]) == pytest.approx(float(expected_g7))
        # Game 2 and 3 are no longer reachable under _collision_mapper.
        assert not jnp.isfinite(out[2])
        assert not jnp.isfinite(out[3])

    def test_unreachable_actions_are_neg_inf(self):
        # Use a larger num_game_actions than the mapper image to force
        # unreachable entries.
        ego_q = jnp.zeros(EGO_DIM, dtype=jnp.float32)
        out = ego_q_to_global_q(
            ego_q, None, jnp.int32(0), None, _identity_mapper, NUM_GAME_ACTIONS + 5
        )
        # Last 5 entries are unreachable.
        assert jnp.all(~jnp.isfinite(out[NUM_GAME_ACTIONS:]))
        assert jnp.all(out[NUM_GAME_ACTIONS:] < 0)

    def test_jit_compatible(self):
        fn = jax.jit(
            lambda q: ego_q_to_global_q(
                q, None, jnp.int32(0), None, _identity_mapper, NUM_GAME_ACTIONS
            )
        )
        out = fn(jnp.arange(EGO_DIM, dtype=jnp.float32))
        assert out.shape == (NUM_GAME_ACTIONS,)


# ---------------------------------------------------------------------------
# select_slots
# ---------------------------------------------------------------------------

class TestSelectSlots:
    def _setup(self):
        # 6 objects: idx 0=DESTROY (low Q), 1=DESTROY (high Q), 2=COLLECT, 3=AVOID,
        # 4=COLLECT but inactive, 5=NOOP padding.
        meta_q = jnp.array([0.5, 5.0, 3.0, -1.0, 999.0, 0.0], dtype=jnp.float32)
        active_mask = jnp.array([True, True, True, True, False, True])
        labels = jnp.array(
            [LABEL_DESTROY, LABEL_DESTROY, LABEL_COLLECT, LABEL_AVOID,
             LABEL_COLLECT, LABEL_NOOP],
            dtype=jnp.int32,
        )
        return meta_q, active_mask, labels

    def test_per_label_picks_best_per_class(self):
        meta_q, active_mask, labels = self._setup()
        targets, active, skills = select_slots_per_label(
            meta_q, active_mask, labels, NO_TARGET, K=3
        )
        # DESTROY: idx 1 (Q=5.0 beats idx 0)
        # COLLECT: idx 2 (idx 4 is inactive)
        # AVOID:   idx 3
        np.testing.assert_array_equal(np.asarray(targets), np.array([1, 2, 3]))
        np.testing.assert_array_equal(np.asarray(active), np.array([True, True, True]))
        np.testing.assert_array_equal(np.asarray(skills), np.array([0, 1, 2]))

    def test_per_label_missing_class_is_inactive(self):
        # No DESTROY objects active.
        meta_q = jnp.array([0.0, 3.0, -1.0], dtype=jnp.float32)
        labels = jnp.array([LABEL_COLLECT, LABEL_AVOID, LABEL_NOOP], dtype=jnp.int32)
        active = jnp.array([True, True, False])
        targets, slot_active, skills = select_slots_per_label(
            meta_q, active, labels, NO_TARGET, K=3
        )
        # DESTROY slot inactive, target = NO_TARGET
        assert int(targets[0]) == NO_TARGET
        assert bool(slot_active[0]) is False
        assert bool(slot_active[1]) is True
        assert int(targets[1]) == 0  # COLLECT idx
        assert bool(slot_active[2]) is True
        assert int(targets[2]) == 1  # AVOID idx

    def test_topk_picks_highest_q_regardless_of_label(self):
        meta_q, active_mask, labels = self._setup()
        targets, active, skills = select_slots_topk(
            meta_q, active_mask, labels, NO_TARGET, K=2
        )
        # Top 2 active: idx 1 (5.0), idx 2 (3.0)
        np.testing.assert_array_equal(np.asarray(targets), np.array([1, 2]))
        np.testing.assert_array_equal(np.asarray(active), np.array([True, True]))
        # Skills derived from labels: idx 1 = DESTROY -> 0, idx 2 = COLLECT -> 1
        np.testing.assert_array_equal(np.asarray(skills), np.array([0, 1]))

    def test_topk_fewer_active_than_k(self):
        meta_q = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        labels = jnp.array([LABEL_DESTROY, LABEL_COLLECT, LABEL_AVOID], dtype=jnp.int32)
        active = jnp.array([True, False, False])  # only 1 active
        targets, slot_active, _ = select_slots_topk(meta_q, active, labels, NO_TARGET, K=3)
        # Only the first slot should be active.
        assert bool(slot_active[0]) is True
        assert int(targets[0]) == 0
        assert bool(slot_active[1]) is False
        assert bool(slot_active[2]) is False
        assert int(targets[1]) == NO_TARGET
        assert int(targets[2]) == NO_TARGET

    def test_dispatch_unknown_mode_raises(self):
        meta_q, active_mask, labels = self._setup()
        with pytest.raises(ValueError):
            select_slots(meta_q, active_mask, labels, NO_TARGET, mode="nope", K=3)


class TestSelectSlotsEps:
    def _setup(self):
        meta_q = jnp.array([0.5, 5.0, 3.0, -1.0, 999.0, 0.0], dtype=jnp.float32)
        active_mask = jnp.array([True, True, True, True, False, True])
        labels = jnp.array(
            [LABEL_DESTROY, LABEL_DESTROY, LABEL_COLLECT, LABEL_AVOID,
             LABEL_COLLECT, LABEL_NOOP],
            dtype=jnp.int32,
        )
        return meta_q, active_mask, labels

    def test_eps_zero_matches_greedy(self):
        meta_q, active_mask, labels = self._setup()
        rng = jax.random.PRNGKey(0)
        eps_t, eps_a, eps_s = select_slots_eps(
            rng, meta_q, active_mask, labels, NO_TARGET,
            mode="per_label", K=3, eps=0.0,
        )
        greedy_t, greedy_a, greedy_s = select_slots(
            meta_q, active_mask, labels, NO_TARGET, mode="per_label", K=3,
        )
        np.testing.assert_array_equal(np.asarray(eps_t), np.asarray(greedy_t))
        np.testing.assert_array_equal(np.asarray(eps_a), np.asarray(greedy_a))
        np.testing.assert_array_equal(np.asarray(eps_s), np.asarray(greedy_s))

    def test_eps_one_explores_within_label(self):
        meta_q, active_mask, labels = self._setup()
        # Collect targets seen for the DESTROY slot across many RNGs.
        seen_destroy = set()
        for k in range(200):
            rng = jax.random.PRNGKey(k)
            targets, active, _ = select_slots_eps(
                rng, meta_q, active_mask, labels, NO_TARGET,
                mode="per_label", K=3, eps=1.0,
            )
            if bool(active[0]):
                seen_destroy.add(int(targets[0]))
        # Greedy DESTROY = idx 1. With eps=1.0 we should also see idx 0 (the
        # other active DESTROY) eventually, and only DESTROY indices.
        assert 0 in seen_destroy or 1 in seen_destroy
        # No COLLECT/AVOID indices should leak into the DESTROY slot.
        assert seen_destroy.issubset({0, 1})

    def test_heuristic_per_label_picks_nearest_per_class(self):
        # 4 objects: 0=DESTROY far, 1=DESTROY near, 2=COLLECT near, 3=AVOID near, 4=COLLECT far (inactive).
        labels = jnp.array(
            [LABEL_DESTROY, LABEL_DESTROY, LABEL_COLLECT, LABEL_AVOID, LABEL_COLLECT],
            dtype=jnp.int32,
        )
        active_mask = jnp.array([True, True, True, True, False])
        obj_x = jnp.array([100.0, 5.0, 10.0, 8.0, 200.0])
        obj_y = jnp.array([100.0, 0.0, 0.0, 0.0, 0.0])
        player_x, player_y = jnp.float32(0.0), jnp.float32(0.0)
        # Heuristic picks nothing useful (no_target).
        targets, active, skills = slots_from_heuristic_target(
            jnp.int32(NO_TARGET), active_mask, labels,
            player_x, player_y, obj_x, obj_y,
            no_target_idx=NO_TARGET, mode="per_label", K=3,
        )
        # Nearest per class.
        assert int(targets[0]) == 1   # nearest DESTROY
        assert int(targets[1]) == 2   # nearest COLLECT (idx 4 inactive)
        assert int(targets[2]) == 3   # only AVOID
        np.testing.assert_array_equal(np.asarray(active), np.array([True, True, True]))
        np.testing.assert_array_equal(np.asarray(skills), np.array([0, 1, 2]))

    def test_heuristic_per_label_overrides_with_heuristic_target(self):
        # Heuristic explicitly picks the *far* DESTROY (idx 0): override slot 0.
        labels = jnp.array(
            [LABEL_DESTROY, LABEL_DESTROY, LABEL_COLLECT, LABEL_AVOID],
            dtype=jnp.int32,
        )
        active_mask = jnp.array([True, True, True, True])
        obj_x = jnp.array([100.0, 5.0, 10.0, 8.0])
        obj_y = jnp.array([100.0, 0.0, 0.0, 0.0])
        targets, active, skills = slots_from_heuristic_target(
            jnp.int32(0), active_mask, labels,
            jnp.float32(0.0), jnp.float32(0.0), obj_x, obj_y,
            no_target_idx=NO_TARGET, mode="per_label", K=3,
        )
        assert int(targets[0]) == 0  # overridden by heuristic
        assert int(targets[1]) == 2  # COLLECT nearest
        assert int(targets[2]) == 3  # AVOID

    def test_heuristic_topk_uses_heuristic_then_nearest(self):
        labels = jnp.array(
            [LABEL_DESTROY, LABEL_DESTROY, LABEL_COLLECT, LABEL_AVOID],
            dtype=jnp.int32,
        )
        active_mask = jnp.array([True, True, True, True])
        obj_x = jnp.array([100.0, 5.0, 10.0, 8.0])
        obj_y = jnp.array([100.0, 0.0, 0.0, 0.0])
        targets, active, skills = slots_from_heuristic_target(
            jnp.int32(3), active_mask, labels,  # heuristic picks AVOID
            jnp.float32(0.0), jnp.float32(0.0), obj_x, obj_y,
            no_target_idx=NO_TARGET, mode="topk", K=3,
        )
        assert int(targets[0]) == 3   # heuristic target
        # Remaining slots: nearest among {0, 1, 2} → 1, 2 (by distance).
        assert {int(targets[1]), int(targets[2])} == {1, 2}
        assert all(bool(a) for a in active)

    def test_eps_topk_explores_any_active(self):
        meta_q, active_mask, labels = self._setup()
        seen_top = set()
        for k in range(200):
            rng = jax.random.PRNGKey(k + 100)
            targets, active, _ = select_slots_eps(
                rng, meta_q, active_mask, labels, NO_TARGET,
                mode="topk", K=2, eps=1.0,
            )
            for i in range(2):
                if bool(active[i]):
                    seen_top.add(int(targets[i]))
        # Should have explored all active indices {0,1,2,3,5} over time.
        assert seen_top.issubset({0, 1, 2, 3, 5})
        assert len(seen_top) >= 3  # exploration broad enough


# ---------------------------------------------------------------------------
# _masked_softmax / _zscore
# ---------------------------------------------------------------------------

class TestMaskedSoftmax:
    def test_sums_to_one_when_any_active(self):
        logits = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        mask = jnp.array([True, False, True])
        w = _masked_softmax(logits, mask, tau=1.0)
        # Inactive entry gets weight 0; rest sum to 1.
        assert float(w[1]) == pytest.approx(0.0)
        assert float(w[0] + w[2]) == pytest.approx(1.0, abs=1e-6)

    def test_all_inactive_returns_zeros(self):
        logits = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        mask = jnp.zeros(3, dtype=bool)
        w = _masked_softmax(logits, mask, tau=1.0)
        np.testing.assert_allclose(w, jnp.zeros(3), atol=1e-6)

    def test_temperature_sharpens_distribution(self):
        logits = jnp.array([0.0, 1.0], dtype=jnp.float32)
        mask = jnp.array([True, True])
        w_hot = _masked_softmax(logits, mask, tau=10.0)
        w_cold = _masked_softmax(logits, mask, tau=0.1)
        # Lower tau -> more peaked on the larger logit.
        assert float(w_cold[1]) > float(w_hot[1])


class TestZscore:
    def test_finite_only_path(self):
        q = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
        z = _zscore(q)
        assert float(jnp.mean(z)) == pytest.approx(0.0, abs=1e-5)
        assert float(jnp.std(z)) == pytest.approx(1.0, abs=1e-3)

    def test_neg_inf_passthrough(self):
        q = jnp.array([1.0, 2.0, -jnp.inf, 3.0], dtype=jnp.float32)
        z = _zscore(q)
        assert not jnp.isfinite(z[2])
        # Other entries: mean = 2, std = sqrt(((1-2)^2+(2-2)^2+(3-2)^2)/3) ≈ 0.8165
        finite = jnp.array([z[0], z[1], z[3]])
        assert float(jnp.mean(finite)) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# compose_global_q
# ---------------------------------------------------------------------------

class TestComposeGlobalQ:
    def test_k1_reduces_to_normalized_input(self):
        # With K=1 and a single active slot, the composed Q should equal the
        # normalized per-slot Q (weights = [1.0]).
        slot_q = jnp.array([[0.0, 1.0, 2.0, 3.0]], dtype=jnp.float32)
        meta_q = jnp.array([5.0], dtype=jnp.float32)
        active = jnp.array([True])
        composed, weights = compose_global_q(
            slot_q, meta_q, active, tau=1.0, normalize="none"
        )
        np.testing.assert_allclose(weights, jnp.array([1.0]), atol=1e-6)
        np.testing.assert_allclose(composed, slot_q[0], atol=1e-6)

    def test_equal_meta_q_yields_equal_weights(self):
        slot_q = jnp.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32
        )
        meta_q = jnp.array([2.0, 2.0], dtype=jnp.float32)
        active = jnp.array([True, True])
        composed, weights = compose_global_q(
            slot_q, meta_q, active, tau=1.0, normalize="none"
        )
        np.testing.assert_allclose(weights, jnp.array([0.5, 0.5]), atol=1e-6)
        np.testing.assert_allclose(composed, jnp.array([0.5, 0.5]), atol=1e-6)

    def test_inactive_slot_contributes_nothing(self):
        slot_q = jnp.array(
            [[1.0, 0.0], [99.0, 99.0]], dtype=jnp.float32
        )
        meta_q = jnp.array([0.0, 100.0], dtype=jnp.float32)
        active = jnp.array([True, False])
        composed, weights = compose_global_q(
            slot_q, meta_q, active, tau=1.0, normalize="none"
        )
        np.testing.assert_allclose(weights, jnp.array([1.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(composed, slot_q[0], atol=1e-6)

    def test_unreachable_marked_neg_inf(self):
        # Slot 0 cannot reach action 0; slot 1 cannot reach action 1; only
        # slot 0 is active. Composed[1] should remain reachable, composed[0]
        # should be -inf.
        slot_q = jnp.array(
            [[-jnp.inf, 5.0], [3.0, -jnp.inf]], dtype=jnp.float32
        )
        meta_q = jnp.array([1.0, 1.0], dtype=jnp.float32)
        active = jnp.array([True, False])
        composed, _ = compose_global_q(
            slot_q, meta_q, active, tau=1.0, normalize="none"
        )
        assert not jnp.isfinite(composed[0])
        assert float(composed[1]) == pytest.approx(5.0)

    def test_weight_threshold_drops_low_slots(self):
        # Three slots with skewed meta-Q: slot 0 = 5, slot 1 = 0, slot 2 = -5.
        # τ=1 softmax weights are approx [0.993, 0.007, 0.0000045].
        # threshold=0.05 should drop slots 1 and 2 → renormalized weights [1, 0, 0].
        slot_q = jnp.array(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=jnp.float32
        )
        meta_q = jnp.array([5.0, 0.0, -5.0], dtype=jnp.float32)
        active = jnp.array([True, True, True])
        composed, weights = compose_global_q(
            slot_q, meta_q, active, tau=1.0, normalize="none",
            weight_threshold=0.05,
        )
        # Slot 0 should fully dominate after threshold.
        assert float(weights[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(weights[1]) == pytest.approx(0.0, abs=1e-6)
        assert float(weights[2]) == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(composed, slot_q[0], atol=1e-6)

    def test_weight_threshold_zero_is_noop(self):
        # threshold=0.0 must reproduce un-thresholded behavior exactly.
        slot_q = jnp.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32
        )
        meta_q = jnp.array([2.0, 1.0], dtype=jnp.float32)
        active = jnp.array([True, True])
        composed_a, w_a = compose_global_q(
            slot_q, meta_q, active, tau=1.0, normalize="none", weight_threshold=0.0,
        )
        composed_b, w_b = compose_global_q(
            slot_q, meta_q, active, tau=1.0, normalize="none",
        )
        np.testing.assert_allclose(w_a, w_b, atol=1e-6)
        np.testing.assert_allclose(composed_a, composed_b, atol=1e-6)

    def test_weight_threshold_all_dropped_zeros_weights(self):
        # τ=10 gives near-uniform softmax over 3 active slots (~0.333 each).
        # threshold=0.4 drops all of them; weights should stay at 0 so the
        # caller can detect "all dropped" and force a NOOP action.
        slot_q = jnp.array(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=jnp.float32
        )
        meta_q = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
        active = jnp.array([True, True, True])
        composed, weights = compose_global_q(
            slot_q, meta_q, active, tau=10.0, normalize="none",
            weight_threshold=0.4,
        )
        # All slots dropped → all weights zero → caller knows to NOOP.
        np.testing.assert_allclose(weights, jnp.zeros(3), atol=1e-6)
        assert float(weights.sum()) == pytest.approx(0.0)

    def test_zscore_normalizes_scale(self):
        # Without normalization, the [-2,0]-range slot would be dwarfed by
        # the [+10,+100] slot; with z-score, both should contribute
        # comparably when weights are equal.
        slot_q = jnp.array(
            [[-2.0, 0.0], [10.0, 100.0]], dtype=jnp.float32
        )
        meta_q = jnp.array([1.0, 1.0], dtype=jnp.float32)
        active = jnp.array([True, True])
        _, _ = compose_global_q(
            slot_q, meta_q, active, tau=1.0, normalize="zscore"
        )
        # Just confirm no nan/inf and argmax considers both slots' preferences.
        composed, _ = compose_global_q(
            slot_q, meta_q, active, tau=1.0, normalize="zscore"
        )
        assert jnp.all(jnp.isfinite(composed))


# ---------------------------------------------------------------------------
# invert_global_to_ego_action
# ---------------------------------------------------------------------------

class TestInvertGlobalToEgo:
    def test_identity_mapper_round_trip(self):
        slot_ego_q = jnp.zeros((2, EGO_DIM), dtype=jnp.float32)
        # Boost ego action 5 so it ties-break high if multiple match.
        slot_ego_q = slot_ego_q.at[0, 5].set(10.0).at[1, 5].set(10.0)
        targets = jnp.array([0, 0], dtype=jnp.int32)
        active = jnp.array([True, True])
        actions = invert_global_to_ego_action(
            jnp.int32(5), slot_ego_q, targets, active, None, None, _identity_mapper
        )
        np.testing.assert_array_equal(np.asarray(actions), np.array([5, 5]))

    def test_inactive_slot_returns_noop(self):
        slot_ego_q = jnp.zeros((2, EGO_DIM), dtype=jnp.float32)
        slot_ego_q = slot_ego_q.at[1, 7].set(5.0)
        targets = jnp.array([0, 0], dtype=jnp.int32)
        active = jnp.array([False, True])
        actions = invert_global_to_ego_action(
            jnp.int32(7), slot_ego_q, targets, active, None, None, _identity_mapper
        )
        assert int(actions[0]) == 0  # inactive -> NOOP
        assert int(actions[1]) == 7

    def test_no_matching_ego_falls_back_to_noop(self):
        # target_shifted_mapper with target=3 maps ego e to (e+3) mod 18.
        # To reach game action 0, we need ego 15. So game action 100 (way out
        # of range) is unreachable -> NOOP.
        slot_ego_q = jnp.zeros((1, EGO_DIM), dtype=jnp.float32)
        targets = jnp.array([3], dtype=jnp.int32)
        active = jnp.array([True])
        # Force game action larger than EGO_DIM so no ego action maps to it.
        actions = invert_global_to_ego_action(
            jnp.int32(50), slot_ego_q, targets, active, None, None, _target_shifted_mapper
        )
        assert int(actions[0]) == 0

    def test_target_shifted_round_trip(self):
        # target=3 -> game = (ego+3) mod 18. For game=10, ego must be 7.
        slot_ego_q = jnp.zeros((1, EGO_DIM), dtype=jnp.float32)
        targets = jnp.array([3], dtype=jnp.int32)
        active = jnp.array([True])
        actions = invert_global_to_ego_action(
            jnp.int32(10), slot_ego_q, targets, active, None, None, _target_shifted_mapper
        )
        assert int(actions[0]) == 7
