"""Tests for stacking train GAME_MODS onto paper_eval OOD keys."""

from concept_based_learning.paper_eval.eval_harness import merge_train_and_eval_game_mods


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
