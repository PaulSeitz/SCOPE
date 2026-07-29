"""Tests for wandb skill-outcome and env-info metric helpers."""

import jax.numpy as jnp
import numpy as np
from flax import struct

from concept_based_learning.observations.egocentric import (
    configure_ego_obs_from_config,
    discrete_ego_flag_layout,
    ego_obs_channels_per_frame,
)
from concept_based_learning.training.logging_metrics import (
    accumulate_flag_combo_counts,
    flag_combination_occupancy_metrics,
    flag_combo_label,
    info_log_metrics,
    metrics_to_wandb_log_dict,
    reset_flag_combo_cumulative,
    resolve_skill_names,
    selected_skill_outcome_metrics,
)


@struct.dataclass
class _FakeInfo:
    successful_rescues: jnp.ndarray
    difficulty: jnp.ndarray


def test_info_log_metrics_struct_dataclass():
    infos = _FakeInfo(
        successful_rescues=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        difficulty=jnp.array([[0.0, 1.0], [1.0, 2.0]]),
    )
    metrics = info_log_metrics(infos)
    assert float(metrics["info/successful_rescues"]) == 2.5
    assert float(metrics["info/difficulty"]) == 1.0


def test_info_log_metrics_dict():
    metrics = info_log_metrics({"score": jnp.array([1.0, 3.0])})
    assert float(metrics["info/score"]) == 2.0


def test_selected_skill_outcome_metrics_separates_success_and_failure():
    # T=2, E=1, S=2 — step0 DESTROY success, step1 COLLECT failure
    active_skill = jnp.array([[0], [1]], dtype=jnp.int32)
    skill_done = jnp.array(
        [[[True, False]], [[False, True]]],
        dtype=jnp.bool_,
    )
    skill_rewards = jnp.array(
        [[[1.0, 0.0]], [[0.0, -1.0]]],
        dtype=jnp.float32,
    )
    metrics = selected_skill_outcome_metrics(
        active_skill,
        skill_done,
        skill_rewards,
        ["DESTROY", "COLLECT"],
    )
    assert float(metrics["skill_DESTROY/selected_frac"]) == 0.5
    assert float(metrics["skill_DESTROY/selected_success_rate"]) == 1.0
    assert float(metrics["skill_DESTROY/selected_failure_rate"]) == 0.0
    assert float(metrics["skill_COLLECT/selected_frac"]) == 0.5
    assert float(metrics["skill_COLLECT/selected_success_rate"]) == 0.0
    assert float(metrics["skill_COLLECT/selected_failure_rate"]) == 1.0
    assert float(metrics["skill/selected_success_frac"]) == 0.5
    assert float(metrics["skill/selected_failure_frac"]) == 0.5


def test_selected_skill_outcome_projects_full_vocab_width():
    """Reward extractors emit 5 channels; metrics must accept that with NUM_SKILLS=3."""
    from concept_based_learning.environments.datatypes import NUM_SKILL_LABELS

    active_skill = jnp.array([[1]], dtype=jnp.int32)  # COLLECT (after NAVIGATE remap)
    # Full vocab: navigate success on channel 3 should fold into COLLECT.
    skill_done = jnp.zeros((1, 1, NUM_SKILL_LABELS), dtype=jnp.bool_).at[..., 3].set(True)
    skill_rewards = jnp.zeros((1, 1, NUM_SKILL_LABELS), dtype=jnp.float32).at[..., 3].set(1.0)
    metrics = selected_skill_outcome_metrics(
        active_skill,
        skill_done,
        skill_rewards,
        ["DESTROY", "COLLECT", "AVOID"],
    )
    assert float(metrics["skill_COLLECT/selected_success_rate"]) == 1.0
    assert float(metrics["skill_COLLECT/selected_success_count"]) == 1.0


def test_project_skill_outcomes_folds_navigate_and_jump():
    from concept_based_learning.environments.datatypes import (
        NUM_SKILL_LABELS,
        project_skill_outcomes,
    )

    rewards = jnp.zeros((NUM_SKILL_LABELS,), dtype=jnp.float32).at[3].set(1.0).at[4].set(-1.0)
    done = jnp.zeros((NUM_SKILL_LABELS,), dtype=jnp.bool_).at[3].set(True).at[4].set(True)
    r3, d3 = project_skill_outcomes(rewards, done, 3)
    assert r3.shape == (3,)
    assert float(r3[1]) == 1.0  # NAVIGATE → COLLECT
    assert float(r3[2]) == -1.0  # JUMP_OVER → AVOID
    assert bool(d3[1]) and bool(d3[2])
    r4, d4 = project_skill_outcomes(rewards, done, 4)
    assert r4.shape == (4,)
    assert float(r4[3]) == 1.0  # NAVIGATE kept
    assert float(r4[2]) == -1.0  # JUMP_OVER → AVOID
    assert bool(d4[3]) and bool(d4[2])
    r5, _ = project_skill_outcomes(rewards, done, 5)
    assert r5.shape == (5,)
    assert float(r5[3]) == 1.0 and float(r5[4]) == -1.0


def test_selected_skill_outcome_respects_valid_mask():
    active_skill = jnp.array([[0]], dtype=jnp.int32)
    skill_done = jnp.array([[[True, False]]], dtype=jnp.bool_)
    skill_rewards = jnp.array([[[1.0, 0.0]]], dtype=jnp.float32)
    metrics = selected_skill_outcome_metrics(
        active_skill,
        skill_done,
        skill_rewards,
        ["DESTROY", "COLLECT"],
        valid_mask=jnp.array([[False]]),
    )
    assert float(metrics["skill_DESTROY/selected_frac"]) == 0.0
    assert float(metrics["skill_DESTROY/selected_success_count"]) == 0.0


def test_resolve_skill_names_defaults():
    names = resolve_skill_names({"NUM_SKILLS": 3}, 3)
    assert names == ["DESTROY", "COLLECT", "AVOID"]


def test_discrete_ego_flag_layout_los_and_weapon_state():
    configure_ego_obs_from_config(
        {
            "USE_LEGACY_EGO_OBS": False,
            "USE_EGO_SIZE_FEATURES": False,
            "USE_EGO_VELOCITY_FEATURES": False,
            "USE_EGO_BLOCKING_FEATURES": True,
            "USE_EGO_WEAPON_FEATURES": True,
            "NUM_SKILLS": 3,
            "ENV_NAME": "seaquest",
        }
    )
    # Base: distance + sin + cos + los = 4; can_fire + facing_enemy = 2; total = 6
    assert ego_obs_channels_per_frame() == 6
    layout = discrete_ego_flag_layout()
    assert layout == [
        ("los_blocked", 3, 2, 0.0),
        ("can_fire", 4, 2, 1.0),
        ("facing_enemy", 5, 2, 1.0),
    ]


def test_flag_combination_occupancy_per_skill_histogram():
    configure_ego_obs_from_config(
        {
            "USE_LEGACY_EGO_OBS": False,
            "USE_EGO_SIZE_FEATURES": False,
            "USE_EGO_VELOCITY_FEATURES": False,
            "USE_EGO_BLOCKING_FEATURES": True,
            "USE_EGO_WEAPON_FEATURES": True,
            "NUM_SKILLS": 2,
            "ENV_NAME": "seaquest",
        }
    )
    cpf = ego_obs_channels_per_frame()
    # Mixed radix: los (2) × can_fire (2) × facing_enemy (2) = 8 cells.
    # DESTROY: los=0, can_fire=+1, facing_enemy=+1
    #   → digits (0, 1, 1) → combo = 0 + 1*2 + 1*2*2 = 6
    # COLLECT: los=1, can_fire=-1, facing_enemy=-1
    #   → digits (1, 0, 0) → combo = 1
    ego0 = jnp.zeros((cpf,), dtype=jnp.float32).at[3].set(0.0).at[4].set(1.0).at[5].set(1.0)
    ego1 = jnp.zeros((cpf,), dtype=jnp.float32).at[3].set(1.0).at[4].set(-1.0).at[5].set(-1.0)
    ego_obs = jnp.stack([ego0, ego1], axis=0)[:, None, :]  # [T=2, E=1, dim]
    active_skill = jnp.array([[0], [1]], dtype=jnp.int32)

    metrics = flag_combination_occupancy_metrics(
        ego_obs, active_skill, ["DESTROY", "COLLECT"]
    )
    assert float(metrics["skill/flag_combo_n_combos"]) == 8.0
    destroy_counts = np.asarray(metrics["skill_DESTROY/flag_combo_counts"])
    collect_counts = np.asarray(metrics["skill_COLLECT/flag_combo_counts"])
    overall = np.asarray(metrics["skill/flag_combo_counts"])
    assert destroy_counts.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert collect_counts.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert overall.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert abs(float(metrics["skill/flag_combo_explored_frac"]) - (2.0 / 8.0)) < 1e-5
    assert abs(float(metrics["skill_DESTROY/flag_combo_explored_frac"]) - (1.0 / 8.0)) < 1e-5
    assert flag_combo_label(6, ["los_blocked", "can_fire", "facing_enemy"], [2, 2, 2]) == (
        "los_blocked=0|can_fire=1|facing_enemy=1"
    )
    assert flag_combo_label(1, ["los_blocked", "can_fire", "facing_enemy"], [2, 2, 2]) == (
        "los_blocked=1|can_fire=-1|facing_enemy=-1"
    )


def test_flag_combo_cumulative_and_wandb_dict():
    reset_flag_combo_cumulative()
    batch = {
        "skill/flag_combo_counts": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "skill_DESTROY/flag_combo_counts": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "update_steps": np.array(1.0),
    }
    cum = accumulate_flag_combo_counts(batch)
    assert abs(float(cum["skill/flag_combo_cum_explored_frac"]) - (1.0 / 6.0)) < 1e-6
    batch2 = {
        "skill/flag_combo_counts": np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "skill_DESTROY/flag_combo_counts": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    cum2 = accumulate_flag_combo_counts(batch2)
    assert float(cum2["skill/flag_combo_cum_explored_frac"]) == 1.0
    log_dict = metrics_to_wandb_log_dict(
        {
            "skill/flag_combo_counts": np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
            "td_loss": 0.5,
        },
        accumulate_flag_combos=False,
    )
    assert log_dict["td_loss"] == 0.5
    assert "skill/flag_combo_hist" in log_dict
    reset_flag_combo_cumulative()


def test_flag_occupancy_empty_when_no_discrete_flags():
    configure_ego_obs_from_config(
        {
            "USE_LEGACY_EGO_OBS": False,
            "USE_EGO_SIZE_FEATURES": False,
            "USE_EGO_VELOCITY_FEATURES": False,
            "USE_EGO_BLOCKING_FEATURES": False,
            "USE_EGO_WEAPON_FEATURES": False,
            "NUM_SKILLS": 3,
            "ENV_NAME": "freeway",
        }
    )
    ego_obs = jnp.zeros((2, 1, ego_obs_channels_per_frame()), dtype=jnp.float32)
    active_skill = jnp.zeros((2, 1), dtype=jnp.int32)
    assert (
        flag_combination_occupancy_metrics(
            ego_obs, active_skill, ["DESTROY", "COLLECT", "AVOID"]
        )
        == {}
    )
