"""Tests for stacking train GAME_MODS onto paper_eval OOD keys."""

from concept_based_learning.paper_eval.eval_harness import (
    _apply_eval_mod_label_overrides,
    merge_train_and_eval_game_mods,
)
from concept_based_learning.paper_eval.mods import ModSpec, mod_specs_for_env


def test_merge_empty():
    assert merge_train_and_eval_game_mods({}, []) is None
    assert merge_train_and_eval_game_mods({"GAME_MODS": None}, ()) is None


def test_merge_keeps_train_mod_on_base():
    cfg = {"GAME_MODS": ["no_surfacing_death"]}
    assert merge_train_and_eval_game_mods(cfg, []) == ["no_surfacing_death"]
    assert merge_train_and_eval_game_mods(cfg, ["faster_enemies"]) == [
        "no_surfacing_death",
        "faster_enemies",
    ]


def test_merge_dedupes():
    cfg = {"GAME_MODS": ["no_surfacing_death"]}
    assert merge_train_and_eval_game_mods(cfg, ["no_surfacing_death"]) == [
        "no_surfacing_death"
    ]


def test_merge_does_not_enable_label_swap():
    """GAME_MODS union must never imply CBL label remapping."""
    cfg = {"GAME_MODS": ["swap_diver_enemy_labels"]}
    merged = merge_train_and_eval_game_mods(cfg, ["lane_scramble"])
    assert "swap_diver_enemy_labels" in merged
    assert "SEAQUEST_SWAP_DIVER_ENEMY_LABELS" not in cfg


def test_relabel_override_only_from_modspec():
    base = {"GAME_MODS": None}
    on = _apply_eval_mod_label_overrides(
        dict(base),
        ModSpec(
            key="swap_diver_enemy_labels",
            mods=("swap_diver_enemy_labels",),
            category="semantic",
            relabel=True,
        ),
    )
    assert on["SEAQUEST_SWAP_DIVER_ENEMY_LABELS"] is True
    assert on["SEAQUEST_SWAP_DIVER_SHARK_ROLES"] is False
    assert on["_PAPER_EVAL_MOD_RELABEL"] is True

    roles = _apply_eval_mod_label_overrides(
        dict(base),
        ModSpec(
            key="swap_diver_shark_roles",
            mods=("swap_diver_shark_roles",),
            category="semantic",
            relabel=True,
        ),
    )
    assert roles["SEAQUEST_SWAP_DIVER_SHARK_ROLES"] is True
    assert roles["SEAQUEST_SWAP_DIVER_ENEMY_LABELS"] is False

    off = _apply_eval_mod_label_overrides(
        dict(base),
        ModSpec(key="base", mods=(), category="control", relabel=False),
    )
    assert off["SEAQUEST_SWAP_DIVER_ENEMY_LABELS"] is False
    assert off["SEAQUEST_SWAP_DIVER_SHARK_ROLES"] is False
    assert off["_PAPER_EVAL_MOD_RELABEL"] is True


def test_seaquest_allowlist_wires_new_keys():
    specs = {s.key: s for s in mod_specs_for_env("seaquest")}
    assert "lane_scramble" in specs
    assert specs["lane_scramble"].mods == ("lane_scramble",)
    assert specs["lane_scramble"].category == "layout"
    assert "dynamic_lane_drift" in specs
    assert specs["dynamic_lane_drift"].category == "layout"
    swap = specs["swap_diver_enemy_labels"]
    assert swap.mods == ("swap_diver_enemy_labels",)
    assert swap.relabel is True
    roles = specs["swap_diver_shark_roles"]
    assert roles.mods == ("swap_diver_shark_roles",)
    assert roles.relabel is True
