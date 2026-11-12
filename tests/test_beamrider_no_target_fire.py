"""Beamrider no-target idle action (FIRE to advance empty sectors)."""

from concept_based_learning.environments.env_manager import (
    EnvManager,
    resolve_no_target_game_action,
)


def test_beamrider_defaults_no_target_to_fire():
    comps = EnvManager().get_components("beamrider")
    assert comps.no_target_game_action == 1
    assert resolve_no_target_game_action({}, comps) == 1
    assert resolve_no_target_game_action(None, comps) == 1


def test_config_overrides_component_default():
    comps = EnvManager().get_components("beamrider")
    assert resolve_no_target_game_action({"NO_TARGET_GAME_ACTION": 0}, comps) == 0


def test_null_config_falls_back_to_component():
    comps = EnvManager().get_components("beamrider")
    assert resolve_no_target_game_action({"NO_TARGET_GAME_ACTION": None}, comps) == 1


def test_seaquest_defaults_no_target_to_noop():
    comps = EnvManager().get_components("seaquest")
    assert comps.no_target_game_action == 0
    assert resolve_no_target_game_action({}, comps) == 0
