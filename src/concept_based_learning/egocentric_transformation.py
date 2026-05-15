# Re-export shim — primary code lives in observations/egocentric.py.
from concept_based_learning.observations.egocentric import (  # noqa: F401
    TARGET_ALIGNED_EGO_OBS_CHANNELS_PER_FRAME,
    LEGACY_EGO_OBS_CHANNELS_PER_FRAME,
    MAX_DIST,
    MAX_SPEED,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    EGO_OBS_CHANNELS_PER_FRAME,
    ego_obs_channels_per_frame,
    set_ego_obs_format,
    using_legacy_ego_obs,
    get_ego_obs,
)