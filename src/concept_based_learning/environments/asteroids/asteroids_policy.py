"""Asteroids heuristic: always select the nearest DESTROY asteroid."""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.datatypes import LABEL_DESTROY
from concept_based_learning.environments.manual_policy import (
    DynamicPick,
    ManualPolicySpec,
    compile_policy,
)
from concept_based_learning.environments.manual_policy.state import NO_TARGET_IDX

ASTEROIDS_POLICY = compile_policy(
    ManualPolicySpec(
        plan=DynamicPick(
            prefer=(LABEL_DESTROY,),
            sticky=False,  # re-rank every step → nearest asteroid
        ),
        no_target_idx=NO_TARGET_IDX,
    )
)

get_heuristic_target = ASTEROIDS_POLICY.as_legacy_heuristic()
get_heuristic_target_with_state = ASTEROIDS_POLICY.as_stateful_heuristic()
