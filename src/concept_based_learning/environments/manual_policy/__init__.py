"""Formalized manual (heuristic) meta-policies.

Python helpers build a small IR (``ManualPolicySpec`` + plan/interrupt nodes)
that compiles to a JAX-traceable ``ManualPolicy.step``.  A future declarative
DSL should emit the same IR; keep nodes data-driven and register named
predicates via ``register_predicate`` instead of baking env logic into the
runtime.
"""

from concept_based_learning.environments.manual_policy.ir import (
    DEFAULT_MISSION_LABELS,
    PREDICATE_REGISTRY,
    ClimbLadders,
    DynamicPick,
    ForceWhen,
    Interrupt,
    ManualPolicySpec,
    NavigateTo,
    Sequence,
    register_predicate,
)
from concept_based_learning.environments.manual_policy.runtime import (
    ManualPolicy,
    compile_policy,
)
from concept_based_learning.environments.manual_policy.state import (
    NO_TARGET_IDX,
    PolicyState,
)

__all__ = [
    "ClimbLadders",
    "DEFAULT_MISSION_LABELS",
    "DynamicPick",
    "ForceWhen",
    "Interrupt",
    "ManualPolicy",
    "ManualPolicySpec",
    "NO_TARGET_IDX",
    "NavigateTo",
    "PREDICATE_REGISTRY",
    "PolicyState",
    "Sequence",
    "compile_policy",
    "register_predicate",
]
