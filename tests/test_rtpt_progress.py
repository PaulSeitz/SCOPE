"""RTPT weighting / unit accumulation (paper_eval mixed ego/pixel + evals)."""

from __future__ import annotations

from concept_based_learning.training import rtpt_progress as rp


class _FakeRTPT:
    def __init__(self) -> None:
        self.n = 0

    def start(self) -> None:
        return None

    def step(self) -> None:
        self.n += 1


def _attach_fake(*, step_every: int = 10, weight: int = 1) -> _FakeRTPT:
    rp.reset_rtpt()
    fake = _FakeRTPT()
    rp._RTPT = fake
    rp._STARTED = True
    rp._STEP_EVERY = step_every
    rp.set_rtpt_job_weight(weight)
    return fake


def test_weight_pixel_skill_and_meta():
    assert rp.rtpt_weight_for_config({"SKILL_INPUT_MODE": "ego"}) == rp.RTPT_WEIGHT_EGO
    assert (
        rp.rtpt_weight_for_config({"SKILL_INPUT_MODE": "pixel_embed"})
        == rp.RTPT_WEIGHT_PIXEL
    )
    assert rp.rtpt_weight_for_config({"SKILL_INPUT_MODE": "oc_embed"}) == rp.RTPT_WEIGHT_OC
    assert (
        rp.rtpt_weight_for_config(
            {"SKILL_INPUT_MODE": "ego", "META_INPUT_MODE": "pixel_embed"},
            kind="tandem",
        )
        == rp.RTPT_WEIGHT_PIXEL
    )


def test_weight_flat_pqn():
    assert (
        rp.rtpt_weight_for_config({"OBJECT_CENTRIC": False}, kind="flat_pqn")
        == rp.RTPT_WEIGHT_PIXEL
    )
    assert (
        rp.rtpt_weight_for_config({"OBJECT_CENTRIC": True}, kind="flat_pqn")
        == rp.RTPT_WEIGHT_OC
    )


def test_pixel_update_consumes_ten_units():
    fake = _attach_fake(step_every=10, weight=rp.RTPT_WEIGHT_PIXEL)
    rp.rtpt_step_callback()
    assert fake.n == 1
    rp.rtpt_step_callback()
    assert fake.n == 2


def test_ego_update_needs_ten_for_one_step():
    fake = _attach_fake(step_every=10, weight=rp.RTPT_WEIGHT_EGO)
    for _ in range(9):
        rp.rtpt_step_callback()
    assert fake.n == 0
    rp.rtpt_step_callback()
    assert fake.n == 1


def test_eval_mod_units_advance_bar():
    fake = _attach_fake(step_every=10, weight=1)
    rp.rtpt_add_units(rp.RTPT_EVAL_UNITS_PER_MOD)  # 25
    assert fake.n == 2
    assert rp._LOGICAL_UPDATES == 5


def test_host_step_still_one_unit():
    fake = _attach_fake(step_every=1, weight=1)
    rp.rtpt_host_step()
    rp.rtpt_host_step()
    assert fake.n == 2


def test_reset_clears_weight():
    _attach_fake(weight=10)
    rp.reset_rtpt()
    assert rp.rtpt_job_weight() == 1
    assert rp._RTPT is None
