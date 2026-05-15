# Re-export shim — primary code lives in heuristic_policy.py.
from concept_based_learning.environments.seaquest.heuristic_policy import (  # noqa: F401
    NO_TARGET_IDX,
    SURFACE_DUMMY_IDX,
    SURFACE_SUB_IDX,
    OXYGEN_LOW_THRESHOLD,
    CRITICAL_OXYGEN_THRESHOLD,
    SURFACE_SUB_NEAR_RANGE,
    CLOSE_DANGER_RANGE,
    DANGER_RELEASE_RANGE,
    X_AXIS_WEIGHT,
    apply_meta_mask_overrides,
    get_forced_target_override,
    get_heuristic_target,
)
