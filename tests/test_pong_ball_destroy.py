"""Pong ball label override (COLLECT vs DESTROY)."""

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import (
    LABEL_COLLECT,
    LABEL_DESTROY,
    configure_label_routing_from_config,
)
from concept_based_learning.environments.pong.pong_label import (
    PongSemanticWrapper,
    configure_pong_ball_label_from_config,
    pong_ball_as_destroy,
    set_pong_ball_as_destroy,
)


class _FakeBall:
    def __init__(self):
        self.x = jnp.array(40.0)
        self.y = jnp.array(50.0)
        self.width = jnp.array(2.0)
        self.height = jnp.array(4.0)


class _FakeObs:
    ball = _FakeBall()


def test_pong_ball_defaults_to_collect():
    set_pong_ball_as_destroy(False)
    wrapper = object.__new__(PongSemanticWrapper)
    wrapper.game_max_objects = 2
    objs = wrapper._extract_frame_objects(_FakeObs())
    assert int(objs.semantic_label[0]) == LABEL_COLLECT


def test_pong_ball_as_destroy_flag():
    set_pong_ball_as_destroy(True)
    try:
        wrapper = object.__new__(PongSemanticWrapper)
        wrapper.game_max_objects = 2
        objs = wrapper._extract_frame_objects(_FakeObs())
        assert int(objs.semantic_label[0]) == LABEL_DESTROY
        assert pong_ball_as_destroy()
    finally:
        set_pong_ball_as_destroy(False)


def test_configure_from_label_routing():
    configure_label_routing_from_config({"PONG_BALL_AS_DESTROY": True})
    assert pong_ball_as_destroy()
    configure_label_routing_from_config({})
    assert not pong_ball_as_destroy()
    configure_pong_ball_label_from_config(None)
    assert not pong_ball_as_destroy()
