"""Egocentric observation features with runtime-selectable format."""

import jax
import jax.numpy as jnp

from concept_based_learning.observations.line_of_sight import line_of_sight_blocked
from concept_based_learning.observations.skill_ego_features import (
    discrete_skill_ego_flags,
    build_skill_ego_extras_stacked,
    configure_ego_weapon_features,
    skill_ego_extra_channels_per_frame,
    using_ego_weapon_features,
)

# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------
# Screen diagonal — used to normalise distance so it stays in [0, 1]
MAX_DIST      = 263.0   # sqrt(160² + 210²)
MAX_SPEED     = 32.0    # pixels/frame, for velocity normalisation
SCREEN_WIDTH  = 160.0
SCREEN_HEIGHT = 210.0

TARGET_ALIGNED_EGO_OBS_CHANNELS_PER_FRAME = 7
LEGACY_EGO_OBS_CHANNELS_PER_FRAME = 10

# Backward-compatible constant used by old imports.
EGO_OBS_CHANNELS_PER_FRAME = TARGET_ALIGNED_EGO_OBS_CHANNELS_PER_FRAME

_USE_LEGACY_EGO_OBS = False
_USE_EGO_SIZE_FEATURES = True
_USE_EGO_VELOCITY_FEATURES = True
# Line-of-sight / blocking channel: 1.0 when an occluding object (BLOCKING_LABELS)
# lies on the player→target segment. Shared by all skills. Off by default.
_USE_EGO_BLOCKING_FEATURES = False
# None = normalize by MAX_DIST (full screen diagonal). Otherwise clip Euclidean
# distance in pixels to this cap, then divide by the cap so feature ∈ [0, 1].
_EGO_DISTANCE_CLIP_MAX_PX: float | None = None
# None / 0 = single normalized distance channel.
# Otherwise replace it with this many triangular, overlapping distance bins.
_EGO_DISTANCE_BINS: int | None = None
# Optional custom centers for distance bins (pixels). If unset and
# _EGO_DISTANCE_BINS is active, centers are generated uniformly in [0, max_px].
_EGO_DISTANCE_BIN_CENTERS_PX: tuple[float, ...] | None = None
# Upper support limit for generated distance bins when centers are not provided.
_EGO_DISTANCE_BIN_MAX_PX: float = MAX_DIST
# When True (and distance bins enabled), collapse distance bins to one scalar
# tier channel in [0,1] (nearest-bin index normalized by n-1).
_EGO_DISTANCE_BINS_SCALAR: bool = False
# When set, replace scalar/bins with one binary channel per threshold (d <= t).
_EGO_DISTANCE_FLAG_THRESHOLDS_PX: tuple[float, ...] | None = None
# None / 0 = keep sin(ψ), cos(ψ) (aim error). Otherwise replace those two
# channels with this many triangular, 50%-overlap bins of world θ_target.
_EGO_ANGLE_BINS: int | None = None
# When True (and angle bins enabled), collapse angle bins to one scalar
# tier channel in [0,1] (nearest-bin index normalized by n-1).
_EGO_ANGLE_BINS_SCALAR: bool = False
# When True (and angle bins enabled), snap world θ to the nearest of ``n``
# hard sectors and emit sin/cos of the sector centre (2 channels, finite
# alphabet on the unit circle). Mutually exclusive with ``_EGO_ANGLE_BINS_SCALAR``.
_EGO_ANGLE_SINCOS_SNAP: bool = False
# POV-only: replace sin/cos(ψ) with overlapping triangular bins on ψ (aim offset).
_EGO_PSI_ANGLE_BINS: int | None = None


def set_ego_obs_format(use_legacy: bool) -> None:
    """Select legacy (10ch) or target-aligned (7ch) ego observation format."""
    global _USE_LEGACY_EGO_OBS
    _USE_LEGACY_EGO_OBS = bool(use_legacy)


def set_ego_size_features(enabled: bool) -> None:
    """Enable/disable size channels while keeping ego feature dimensions unchanged."""
    global _USE_EGO_SIZE_FEATURES
    _USE_EGO_SIZE_FEATURES = bool(enabled)


def set_ego_velocity_features(enabled: bool) -> None:
    """Enable/disable velocity channels while keeping ego feature dimensions unchanged."""
    global _USE_EGO_VELOCITY_FEATURES
    _USE_EGO_VELOCITY_FEATURES = bool(enabled)


def using_ego_size_features() -> bool:
    """Return True when target/player size channels are included in ego features."""
    return _USE_EGO_SIZE_FEATURES


def using_ego_velocity_features() -> bool:
    """Return True when radial/tangential velocity channels are included."""
    return _USE_EGO_VELOCITY_FEATURES


def set_ego_blocking_features(enabled: bool) -> None:
    """Enable/disable the shared line-of-sight blocked channel (adds 1 channel/frame)."""
    global _USE_EGO_BLOCKING_FEATURES
    _USE_EGO_BLOCKING_FEATURES = bool(enabled)


def set_ego_los_features(enabled: bool) -> None:
    """Backward-compatible alias for ``set_ego_blocking_features``."""
    set_ego_blocking_features(enabled)


def using_ego_blocking_features() -> bool:
    """Return True when the shared line-of-sight blocked channel is included."""
    return _USE_EGO_BLOCKING_FEATURES


def using_ego_los_features() -> bool:
    """Backward-compatible alias for ``using_ego_blocking_features``."""
    return using_ego_blocking_features()


def using_legacy_ego_obs() -> bool:
    """Return True when the legacy ego observation format is active."""
    return _USE_LEGACY_EGO_OBS


def set_ego_distance_clip_max_pixels(max_px: float | None) -> None:
    """
    Clip target-aligned Euclidean distance before normalization.

    When set to a positive value (e.g. 50), distances >= max_px are treated as
    max_px so the network sees a single \"far\" bucket; normalization uses max_px.
    Pass ``None`` to disable clipping (normalize by screen diagonal ``MAX_DIST``).
    Pass ``0`` to **zero the distance channel** (angle-only ablation; input dim
    unchanged). This is *not* \"clip at 0px\" — that would saturate everything to
    contact.
    """
    global _EGO_DISTANCE_CLIP_MAX_PX
    if max_px is None:
        _EGO_DISTANCE_CLIP_MAX_PX = None
        return
    cap = float(max_px)
    if cap < 0:
        raise ValueError(f"EGO_DISTANCE_CLIP_MAX_PX must be >= 0, got {max_px}")
    _EGO_DISTANCE_CLIP_MAX_PX = cap


def ego_distance_clip_max_pixels() -> float | None:
    """Return the active distance clip cap in pixels, or None if disabled."""
    return _EGO_DISTANCE_CLIP_MAX_PX


def ego_distance_zeroed() -> bool:
    """True when distance features are forced to 0 (``EGO_DISTANCE_CLIP_MAX_PX: 0``)."""
    return _EGO_DISTANCE_CLIP_MAX_PX is not None and float(_EGO_DISTANCE_CLIP_MAX_PX) == 0.0


def set_ego_distance_bins(n_bins: int | None) -> None:
    """Replace scalar distance with ``n_bins`` overlapping triangular bins."""
    global _EGO_DISTANCE_BINS
    if n_bins is None:
        _EGO_DISTANCE_BINS = None
        return
    n = int(n_bins)
    if n <= 0:
        _EGO_DISTANCE_BINS = None
        return
    if n < 2:
        raise ValueError(f"EGO_DISTANCE_BINS must be >= 2, got {n_bins}")
    _EGO_DISTANCE_BINS = n


def set_ego_distance_bins_scalar(enabled: bool) -> None:
    """Use one scalar tier channel instead of N distance-bin channels."""
    global _EGO_DISTANCE_BINS_SCALAR
    _EGO_DISTANCE_BINS_SCALAR = bool(enabled)


def ego_distance_bins() -> int | None:
    """Return active distance-bin count, or None for scalar distance."""
    return _EGO_DISTANCE_BINS


def set_ego_distance_flag_thresholds_px(
    thresholds: list[float] | tuple[float, ...] | None,
) -> None:
    """Replace distance scalar/bins with nested ``d <= threshold`` indicator channels."""
    global _EGO_DISTANCE_FLAG_THRESHOLDS_PX
    if thresholds is None:
        _EGO_DISTANCE_FLAG_THRESHOLDS_PX = None
        return
    vals = tuple(float(t) for t in thresholds)
    if len(vals) < 1:
        raise ValueError("EGO_DISTANCE_FLAG_THRESHOLDS_PX must have at least one threshold")
    if any(v < 0.0 for v in vals):
        raise ValueError("EGO_DISTANCE_FLAG_THRESHOLDS_PX values must be >= 0")
    if any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
        raise ValueError("EGO_DISTANCE_FLAG_THRESHOLDS_PX must be strictly increasing")
    _EGO_DISTANCE_FLAG_THRESHOLDS_PX = vals


def ego_distance_flag_thresholds_px() -> tuple[float, ...] | None:
    return _EGO_DISTANCE_FLAG_THRESHOLDS_PX


def set_ego_distance_bin_centers_px(centers: list[float] | tuple[float, ...] | None) -> None:
    """Set explicit distance-bin centers (pixels), or None for autogenerated centers."""
    global _EGO_DISTANCE_BIN_CENTERS_PX
    if centers is None:
        _EGO_DISTANCE_BIN_CENTERS_PX = None
        return
    vals = tuple(float(c) for c in centers)
    if len(vals) < 2:
        raise ValueError("EGO_DISTANCE_BIN_CENTERS_PX must have at least 2 centers")
    if any(c < 0.0 for c in vals):
        raise ValueError("EGO_DISTANCE_BIN_CENTERS_PX values must be >= 0")
    if any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
        raise ValueError("EGO_DISTANCE_BIN_CENTERS_PX must be strictly increasing")
    _EGO_DISTANCE_BIN_CENTERS_PX = vals


def set_ego_distance_bin_max_px(max_px: float | None) -> None:
    """Set max distance support used for autogenerated bin centers."""
    global _EGO_DISTANCE_BIN_MAX_PX
    if max_px is None:
        _EGO_DISTANCE_BIN_MAX_PX = MAX_DIST
        return
    cap = float(max_px)
    if cap <= 0:
        raise ValueError(f"EGO_DISTANCE_BIN_MAX_PX must be positive, got {max_px}")
    _EGO_DISTANCE_BIN_MAX_PX = cap


def set_ego_angle_bins(n_bins: int | None) -> None:
    """Replace sin/cos(ψ) with ``n_bins`` overlapping world-θ bins.

    ``None`` / ``0`` restores continuous sin/cos(ψ). ``n_bins`` must be >= 2.
    Bins are triangular with 50% overlap and wrap at ±π (see
    :func:`triangular_angle_bins`).
    """
    global _EGO_ANGLE_BINS
    if n_bins is None:
        _EGO_ANGLE_BINS = None
        return
    n = int(n_bins)
    if n <= 0:
        _EGO_ANGLE_BINS = None
        return
    if n < 2:
        raise ValueError(f"EGO_ANGLE_BINS must be >= 2, got {n_bins}")
    _EGO_ANGLE_BINS = n


def set_ego_psi_angle_bins(n_bins: int | None) -> None:
    """POV-only: replace sin/cos(ψ) with ``n_bins`` overlapping ψ bins."""
    global _EGO_PSI_ANGLE_BINS
    if n_bins is None:
        _EGO_PSI_ANGLE_BINS = None
        return
    n = int(n_bins)
    if n <= 0:
        _EGO_PSI_ANGLE_BINS = None
        return
    if n < 2:
        raise ValueError(f"EGO_PSI_ANGLE_BINS must be >= 2, got {n_bins}")
    _EGO_PSI_ANGLE_BINS = n


def ego_psi_angle_bins() -> int | None:
    return _EGO_PSI_ANGLE_BINS


def set_ego_angle_bins_scalar(enabled: bool) -> None:
    """Use one scalar tier channel instead of N angle-bin channels."""
    global _EGO_ANGLE_BINS_SCALAR
    _EGO_ANGLE_BINS_SCALAR = bool(enabled)


def set_ego_angle_sincos_snap(enabled: bool) -> None:
    """Snap world θ to hard sector centres; emit sin/cos (2 channels)."""
    global _EGO_ANGLE_SINCOS_SNAP
    _EGO_ANGLE_SINCOS_SNAP = bool(enabled)


def ego_angle_bins() -> int | None:
    """Return the active world-θ bin count, or None when using sin/cos(ψ)."""
    return _EGO_ANGLE_BINS


def ego_angle_channels_per_frame() -> int:
    """Pose angle block width: ``n`` bins, or 2 for sin/cos(ψ)."""
    from concept_based_learning.observations.pov_relative import using_pov_relative

    if using_pov_relative() and _EGO_PSI_ANGLE_BINS:
        return int(_EGO_PSI_ANGLE_BINS)
    if _EGO_ANGLE_BINS:
        if _EGO_ANGLE_SINCOS_SNAP:
            return 2
        return 1 if _EGO_ANGLE_BINS_SCALAR else int(_EGO_ANGLE_BINS)
    return 2


def quantize_angle_sincos(theta: jnp.ndarray, n_bins: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Hard world-θ sectors: sin/cos of the sector centre (``n_bins`` levels)."""
    n = int(n_bins)
    t = (jnp.asarray(theta, dtype=jnp.float32) + jnp.pi) / (jnp.float32(2.0) * jnp.pi)
    idx = (jnp.floor(t * n).astype(jnp.int32) % n).astype(jnp.float32)
    center = (idx + 0.5) * (jnp.float32(2.0) * jnp.pi) / n - jnp.pi
    return jnp.sin(center), jnp.cos(center)


def triangular_angle_bins(theta: jnp.ndarray, n_bins: int) -> jnp.ndarray:
    """Triangular circular bins of ``theta`` (radians), 50% overlap.

    Centres at ``k * 2π / n`` (k=0 is +x / screen-right). Half-width is
    ``2π / n`` so adjacent bins are 0.5 at the midpoint and the features
    form a partition of unity. Output shape is ``theta.shape + (n_bins,)``.
    """
    n = int(n_bins)
    centers = jnp.arange(n, dtype=jnp.float32) * (jnp.float32(2.0 * jnp.pi) / n)
    delta = jnp.asarray(theta, dtype=jnp.float32)[..., None] - centers
    delta = jnp.arctan2(jnp.sin(delta), jnp.cos(delta))
    half = jnp.float32(2.0 * jnp.pi) / n
    return jnp.maximum(0.0, 1.0 - jnp.abs(delta) / half)


_USE_DESTROY_ONLY_EGO_EXTRAS = False


def set_destroy_only_ego_extras(enabled: bool) -> None:
    """When True, LoS/weapon extras are concatenated only for the DESTROY head."""
    global _USE_DESTROY_ONLY_EGO_EXTRAS
    _USE_DESTROY_ONLY_EGO_EXTRAS = bool(enabled)


def using_destroy_only_ego_extras() -> bool:
    return _USE_DESTROY_ONLY_EGO_EXTRAS


def configure_ego_obs_from_config(config: dict) -> None:
    """Apply all ego observation toggles from a merged experiment config dict."""
    from concept_based_learning.observations.pov_relative import (
        configure_pov_relative_from_config,
        using_pov_relative,
    )
    from concept_based_learning.observations.pov_fixed_bilateral import (
        configure_pov_fixed_bilateral_from_config,
    )
    from concept_based_learning.observations.pov_turn import (
        configure_pov_turn_from_config,
    )

    configure_pov_relative_from_config(config)
    configure_pov_fixed_bilateral_from_config(config)
    configure_pov_turn_from_config(config)
    set_ego_obs_format(bool(config.get("USE_LEGACY_EGO_OBS", False)))
    set_ego_size_features(bool(config.get("USE_EGO_SIZE_FEATURES", True)))
    set_ego_velocity_features(bool(config.get("USE_EGO_VELOCITY_FEATURES", True)))
    set_destroy_only_ego_extras(bool(config.get("USE_DESTROY_ONLY_EGO_EXTRAS", False)))
    # Shared blocking/LOS flag (new name preferred; old USE_EGO_LOS_FEATURES still works).
    blocking = config.get(
        "USE_EGO_BLOCKING_FEATURES",
        config.get("USE_EGO_LOS_FEATURES", False),
    )
    set_ego_blocking_features(bool(blocking))
    flag_thresh = config.get("EGO_DISTANCE_FLAG_THRESHOLDS_PX", None)
    if flag_thresh not in (None, "", False):
        set_ego_distance_flag_thresholds_px(flag_thresh)
        set_ego_distance_clip_max_pixels(None)
        set_ego_distance_bins(None)
        set_ego_distance_bins_scalar(False)
        set_ego_distance_bin_centers_px(None)
    else:
        set_ego_distance_flag_thresholds_px(None)
        clip_px = config.get("EGO_DISTANCE_CLIP_MAX_PX", None)
        set_ego_distance_clip_max_pixels(None if clip_px is None else float(clip_px))
        dist_bins = config.get("EGO_DISTANCE_BINS", None)
        set_ego_distance_bins(None if dist_bins in (None, "", False) else int(dist_bins))
        set_ego_distance_bins_scalar(bool(config.get("EGO_DISTANCE_BINS_SCALAR", False)))
        dist_centers = config.get("EGO_DISTANCE_BIN_CENTERS_PX", None)
        set_ego_distance_bin_centers_px(dist_centers)
        set_ego_distance_bin_max_px(config.get("EGO_DISTANCE_BIN_MAX_PX", MAX_DIST))
    angle_bins = config.get("EGO_ANGLE_BINS", None)
    psi_bins = config.get("EGO_PSI_ANGLE_BINS", None)
    if using_pov_relative():
        set_ego_psi_angle_bins(
            None if psi_bins in (None, "", False) else int(psi_bins)
        )
        angle_bins = None
    else:
        set_ego_psi_angle_bins(None)
    set_ego_angle_bins(None if angle_bins in (None, "", False) else int(angle_bins))
    set_ego_angle_sincos_snap(bool(config.get("EGO_ANGLE_SINCOS_SNAP", False)))
    set_ego_angle_bins_scalar(
        bool(config.get("EGO_ANGLE_BINS_SCALAR", False))
        and not bool(config.get("EGO_ANGLE_SINCOS_SNAP", False))
    )
    # Weapon/shooting channel (new name preferred; old USE_SKILL_EGO_FEATURES still works).
    # Channel count is fixed (1) when enabled — no separate EXTRA_DIM config.
    weapon = config.get(
        "USE_EGO_WEAPON_FEATURES",
        config.get("USE_SKILL_EGO_FEATURES", False),
    )
    configure_ego_weapon_features(
        enabled=bool(weapon),
        env_name=str(config.get("ENV_NAME", "")),
        shot_features=bool(config.get("USE_EGO_SHOT_FEATURES", False)),
        shot_between=bool(config.get("USE_EGO_SHOT_BETWEEN", False)),
    )


def distance_feature_from_pixels(distances_px: jnp.ndarray) -> jnp.ndarray:
    """Map raw pixel distances to the normalized distance channel fed to skills."""
    if _EGO_DISTANCE_CLIP_MAX_PX is not None:
        cap = jnp.float32(_EGO_DISTANCE_CLIP_MAX_PX)
        # Sentinel 0: drop distance information (keep channel width for arch parity).
        return jnp.where(
            cap == jnp.float32(0.0),
            jnp.zeros_like(distances_px, dtype=jnp.float32),
            jnp.minimum(distances_px, cap) / cap,
        )
    return distances_px / MAX_DIST


def ego_distance_channels_per_frame() -> int:
    """Distance block width: flag indicators, ``n`` bins, or 1 scalar channel."""
    if _EGO_DISTANCE_FLAG_THRESHOLDS_PX:
        return len(_EGO_DISTANCE_FLAG_THRESHOLDS_PX)
    if _EGO_DISTANCE_BINS:
        return 1 if _EGO_DISTANCE_BINS_SCALAR else int(_EGO_DISTANCE_BINS)
    return 1


def triangular_distance_bins(distances_px: jnp.ndarray, centers_px: jnp.ndarray) -> jnp.ndarray:
    """Triangular bins over distance (pixels), with overlap from neighboring centers."""
    d = jnp.asarray(distances_px, dtype=jnp.float32)[..., None]
    centers = jnp.asarray(centers_px, dtype=jnp.float32)
    n = int(centers.shape[0])
    left_w = jnp.zeros_like(centers)
    right_w = jnp.zeros_like(centers)
    left_w = left_w.at[1:].set(centers[1:] - centers[:-1])
    right_w = right_w.at[:-1].set(centers[1:] - centers[:-1])
    # End bins keep support to the nearest neighbor spacing.
    left_w = left_w.at[0].set(right_w[0])
    right_w = right_w.at[-1].set(left_w[-1])
    left = centers - left_w
    right = centers + right_w
    up = (d - left) / jnp.maximum(centers - left, 1e-6)
    down = (right - d) / jnp.maximum(right - centers, 1e-6)
    tri = jnp.minimum(up, down)
    tri = jnp.maximum(tri, 0.0)
    # Degenerate safety fallback.
    if n == 1:
        return jnp.ones_like(tri)
    return tri


def distance_features_from_pixels(distances_px: jnp.ndarray) -> list[jnp.ndarray]:
    """Distance feature block as list of per-channel vectors."""
    if _EGO_DISTANCE_FLAG_THRESHOLDS_PX:
        thresholds = jnp.asarray(_EGO_DISTANCE_FLAG_THRESHOLDS_PX, dtype=jnp.float32)
        d = jnp.asarray(distances_px, dtype=jnp.float32)[..., None]
        flags = (d <= thresholds).astype(jnp.float32)
        return [flags[..., i] for i in range(int(flags.shape[-1]))]
    if _EGO_DISTANCE_BINS:
        if _EGO_DISTANCE_BIN_CENTERS_PX is not None:
            centers = jnp.asarray(_EGO_DISTANCE_BIN_CENTERS_PX, dtype=jnp.float32)
        else:
            max_px = jnp.float32(_EGO_DISTANCE_BIN_MAX_PX)
            centers = jnp.linspace(
                jnp.float32(0.0),
                max_px,
                int(_EGO_DISTANCE_BINS),
                dtype=jnp.float32,
            )
        bin_feats = triangular_distance_bins(distances_px, centers)
        if _EGO_DISTANCE_BINS_SCALAR:
            n = jnp.float32(bin_feats.shape[-1] - 1)
            idx = jnp.argmax(bin_feats, axis=-1).astype(jnp.float32)
            scalar = jnp.where(n > 0, idx / n, jnp.zeros_like(idx))
            return [scalar]
        return [bin_feats[:, i] for i in range(int(bin_feats.shape[-1]))]
    return [distance_feature_from_pixels(distances_px)]


def distance_pixels_from_feature(feature: jnp.ndarray | float) -> float:
    """Invert the distance channel for logging / overlays (clipped view)."""
    if _EGO_DISTANCE_CLIP_MAX_PX is not None:
        if float(_EGO_DISTANCE_CLIP_MAX_PX) == 0.0:
            return 0.0
        return float(feature) * _EGO_DISTANCE_CLIP_MAX_PX
    return float(feature) * MAX_DIST


def _ego_velocity_channels_per_frame() -> int:
    """POV uses one signed view-radial approach channel; otherwise radial+tangential."""
    if not _USE_EGO_VELOCITY_FEATURES:
        return 0
    from concept_based_learning.observations.pov_fixed_bilateral import (
        using_pov_fixed_bilateral,
    )
    from concept_based_learning.observations.pov_relative import using_pov_relative

    if using_pov_fixed_bilateral():
        return 0
    return 1 if using_pov_relative() else 2


def ego_obs_channels_per_frame() -> int:
    """Return active per-frame ego feature channel count."""
    los_channels = 1 if _USE_EGO_BLOCKING_FEATURES else 0
    if _USE_LEGACY_EGO_OBS:
        # Legacy channel groups:
        #   always: dx, dy, sin(o), cos(o)             -> 4
        #   velocity (optional): dvx, dvy              -> +2
        #   size (optional): t_w, t_h, p_w, p_h        -> +4
        #   line-of-sight (optional): blocked          -> +1
        return 4 + (2 if _USE_EGO_VELOCITY_FEATURES else 0) + (
            4 if _USE_EGO_SIZE_FEATURES else 0
        ) + los_channels + skill_ego_extra_channels_per_frame()
    # Target-aligned channel groups:
    #   always: distance + angle (sin/cos(ψ) or N θ bins)
    #   velocity (optional): POV approach (1) or radial+tangential (2)
    #   size (optional): target_w, target_h           -> +2
    #   line-of-sight (optional): blocked             -> +1
    pose = ego_distance_channels_per_frame() + ego_angle_channels_per_frame()
    return pose + _ego_velocity_channels_per_frame() + (
        2 if _USE_EGO_SIZE_FEATURES else 0
    ) + los_channels + skill_ego_extra_channels_per_frame()


def ego_base_channels_per_frame() -> int:
    """Per-frame channel count excluding skill extras (includes LOS when enabled)."""
    return ego_obs_channels_per_frame() - skill_ego_extra_channels_per_frame()


def ego_pose_channels_per_frame() -> int:
    """Pose block shared by every skill: distance (or dx/dy), optional vel/size, angle.

    Angle is sin/cos(ψ) (2) or ``EGO_ANGLE_BINS`` overlapping world-θ bins.
    LoS and weapon extras are *not* included; those are DESTROY concatenations
    when ``USE_DESTROY_ONLY_EGO_EXTRAS`` is on (and still a shared suffix otherwise).
    Pose is a prefix of the full ``get_ego_obs`` vector.
    """
    if _USE_LEGACY_EGO_OBS:
        return 4 + (2 if _USE_EGO_VELOCITY_FEATURES else 0) + (
            4 if _USE_EGO_SIZE_FEATURES else 0
        )
    return ego_distance_channels_per_frame() + ego_angle_channels_per_frame() + (
        _ego_velocity_channels_per_frame()
    ) + (2 if _USE_EGO_SIZE_FEATURES else 0)


def ego_destroy_extra_channels_per_frame() -> int:
    """LoS + weapon extras appended after the pose prefix."""
    los_channels = 1 if _USE_EGO_BLOCKING_FEATURES else 0
    return los_channels + skill_ego_extra_channels_per_frame()


def skill_ego_flat_dim(
    skill_idx: int,
    *,
    stack_size: int,
    destroy_skill_idx: int = 0,
) -> int:
    """Flattened ego width for one skill head (pose × stack, plus DESTROY extras)."""
    pose = ego_pose_channels_per_frame() * int(stack_size)
    extra = ego_destroy_extra_channels_per_frame() * int(stack_size)
    if (
        using_destroy_only_ego_extras()
        and extra > 0
        and int(skill_idx) != int(destroy_skill_idx)
    ):
        return pose
    return pose + extra


def take_ego_pose(ego: jnp.ndarray, *, stack_size: int) -> jnp.ndarray:
    """Keep the pose prefix of each stacked frame (not a flat leading slice).

    ``get_ego_obs`` flattens ``[stack, channels]`` so extras sit *inside* every
    frame, after the pose channels. ``x[..., :12]`` would keep frame-0 extras
    and drop later pose.
    """
    full_ch = ego_obs_channels_per_frame()
    pose_ch = ego_pose_channels_per_frame()
    stack = int(stack_size)
    if ego.shape[-1] != stack * full_ch:
        raise ValueError(
            f"ego tail dim {ego.shape[-1]} != stack {stack} × channels {full_ch}"
        )
    leading = ego.shape[:-1]
    framed = ego.reshape(*leading, stack, full_ch)
    return framed[..., :pose_ch].reshape(*leading, pose_ch * stack)


def slice_skill_ego(
    ego: jnp.ndarray,
    skill_idx: int,
    *,
    stack_size: int,
    destroy_skill_idx: int = 0,
) -> jnp.ndarray:
    """Slice a full ``get_ego_obs`` vector to the head's input width.

    ``skill_idx`` must be a Python int so the branch is static under JIT.
    """
    width = skill_ego_flat_dim(
        skill_idx, stack_size=stack_size, destroy_skill_idx=destroy_skill_idx
    )
    if width == ego.shape[-1]:
        return ego
    return take_ego_pose(ego, stack_size=stack_size)


def destroy_only_extras_active() -> bool:
    """True when non-DESTROY heads should drop the LoS/weapon suffix."""
    return using_destroy_only_ego_extras() and ego_destroy_extra_channels_per_frame() > 0


def discrete_ego_flag_layout() -> list[tuple[str, int, int, float]]:
    """
    Active discrete flag channels in the latest ego frame.

    Returns ``(name, channel_index_within_frame, n_levels, level_offset)`` in
    digit order used by occupancy histograms (digit 0 = first entry). Shared
    flags (LOS) come first; per-skill extras follow in registry order.
    Binary flags use ``n_levels=2`` (los, can_fire, facing_enemy).
    """
    layout: list[tuple[str, int, int, float]] = []
    extras_width = skill_ego_extra_channels_per_frame()
    base = ego_base_channels_per_frame()
    if _USE_EGO_BLOCKING_FEATURES:
        # LOS is the last base channel (immediately before skill extras).
        layout.append(("los_blocked", base - 1, 2, 0.0))
    if using_ego_weapon_features() and extras_width > 0:
        for flag in discrete_skill_ego_flags():
            idx = base + int(flag.offset)
            if 0 <= flag.offset < extras_width:
                layout.append(
                    (flag.name, idx, int(flag.n_levels), float(flag.level_offset))
                )
    return layout


# ---------------------------------------------------------------------------
# Player entity accessor (supports different field names across games)
# ---------------------------------------------------------------------------

def _get_player(raw_observation):
    """Return the player entity from a raw observation.

    Most games expose ``raw_observation.player``; Freeway uses
    ``raw_observation.chicken``.  Additional aliases can be added here
    without touching the rest of the module.
    """
    for name in ("player", "chicken", "agent", "skier"):
        if hasattr(raw_observation, name):
            return getattr(raw_observation, name)
    raise AttributeError(
        f"Cannot find a player entity in {type(raw_observation).__name__}. "
        "Expected one of: player, chicken, agent, skier."
    )


def _uses_bottom_edge_y(raw_observation) -> bool:
    """
    Return True when this environment stores bbox y as a bottom edge.

    Freeway (``raw_observation.chicken``) is the main case today. Most other games
    expose ``raw_observation.player`` with top-left y semantics.
    """
    return hasattr(raw_observation, "chicken")


def _center_y(y: jnp.ndarray, height: jnp.ndarray, *, y_is_bottom_edge: bool) -> jnp.ndarray:
    """
    Compute bbox centre y under an explicit coordinate convention.

    - top-left convention:    center_y = y + h / 2
    - bottom-edge convention: center_y = y - h / 2
    """
    if y_is_bottom_edge:
        return y - height / 2.0
    return y + height / 2.0


# ---------------------------------------------------------------------------
# Per-frame geometry (vmapped over the frame stack)
# ---------------------------------------------------------------------------

def _box_surface_extent(half_w: jnp.ndarray, half_h: jnp.ndarray, cos_t: jnp.ndarray, sin_t: jnp.ndarray) -> jnp.ndarray:
    """
    Distance from box centre to the surface point hit by ray (cos_t, sin_t).

    Clips the unit ray to the axis-aligned box [−hw, hw] × [−hh, hh], giving the
    exact nearest surface point — a corner when both axes are simultaneously hit,
    an edge midpoint when only one axis is hit.  Fully differentiable; no branches.
    """
    # How far along the ray until each axis boundary is hit.
    # jnp.where guards against division by zero on axis-aligned rays.
    t_x = jnp.where(jnp.abs(cos_t) > 1e-8, half_w / jnp.abs(cos_t), jnp.inf)
    t_y = jnp.where(jnp.abs(sin_t) > 1e-8, half_h / jnp.abs(sin_t), jnp.inf)
    t_exit = jnp.minimum(t_x, t_y)
    # Euclidean length of the clipped ray segment
    return jnp.sqrt((t_exit * cos_t) ** 2 + (t_exit * sin_t) ** 2)


def aabb_surface_gap_px(
    p_x,
    p_y,
    p_w,
    p_h,
    t_x,
    t_y,
    t_w,
    t_h,
    *,
    y_is_bottom_edge: bool = False,
):
    """
    Nearest-surface gap between two AABBs (same geometry as skill ego distance).

    Centre→centre ray, minus each box's surface extent along that ray; clamped
    at 0 when boxes overlap/touch. All args broadcast (scalars vs per-object).
    """
    p_cx = p_x + p_w / 2.0
    p_cy = _center_y(p_y, p_h, y_is_bottom_edge=y_is_bottom_edge)
    t_cx = t_x + t_w / 2.0
    t_cy = _center_y(t_y, t_h, y_is_bottom_edge=y_is_bottom_edge)

    dx = t_cx - p_cx
    dy = t_cy - p_cy
    center_dist = jnp.sqrt(dx ** 2 + dy ** 2 + 1e-6)
    theta = jnp.arctan2(dy, dx)
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    p_extent = _box_surface_extent(p_w / 2.0, p_h / 2.0, cos_t, sin_t)
    t_extent = _box_surface_extent(t_w / 2.0, t_h / 2.0, cos_t, sin_t)
    return jnp.maximum(center_dist - p_extent - t_extent, 0.0)


def player_object_aabb_gaps(raw_observation, objects, *, y_is_bottom_edge=None):
    """
    Per-object AABB surface gaps from the controllable agent to ``objects``.

    Returns ``(distances, p_cx, p_cy, o_cx, o_cy)`` using the last stacked frame.
    ``y_is_bottom_edge`` defaults to :func:`_uses_bottom_edge_y`.
    """
    if y_is_bottom_edge is None:
        y_is_bottom_edge = _uses_bottom_edge_y(raw_observation)
    player = _get_player(raw_observation)
    px = player.x[-1]
    py = player.y[-1]
    pw = player.width[-1]
    ph = player.height[-1]
    ox = objects.x[-1]
    oy = objects.y[-1]
    ow = objects.width[-1]
    oh = objects.height[-1]
    distances = aabb_surface_gap_px(
        px, py, pw, ph, ox, oy, ow, oh, y_is_bottom_edge=y_is_bottom_edge
    )
    p_cx = px + pw / 2.0
    p_cy = _center_y(py, ph, y_is_bottom_edge=y_is_bottom_edge)
    o_cx = ox + ow / 2.0
    o_cy = _center_y(oy, oh, y_is_bottom_edge=y_is_bottom_edge)
    return distances, p_cx, p_cy, o_cx, o_cy


def _frame_geometry_target_aligned(
    p_x,
    p_y,
    p_w,
    p_h,
    p_ori_deg,
    t_x,
    t_y,
    t_w,
    t_h,
    *,
    y_is_bottom_edge: bool,
):
    """
    Compute target-aligned geometry for a single frame in the observation stack.

    All inputs are scalars (a single frame slice); this function is vmapped over
    the stack axis by get_ego_obs.

    Returns
    -------
    distance      : nearest-surface gap between player and target bounding boxes
    theta_target  : angle of player→target vector (centre–centre), screen-Y-down radians
    psi           : player facing minus theta_target, wrapped to [-π, π]
    t_cx, t_cy    : target centre coords (used for cross-frame velocity)
    p_cx, p_cy    : player centre coords (used for cross-frame velocity)
    t_w, t_h      : target bounding-box sizes (passed through)
    """
    # IMPORTANT: centre computation must match the environment's y convention.
    # If this drifts from action mapping/reward geometry, transfer can silently fail.
    distance = aabb_surface_gap_px(
        p_x,
        p_y,
        p_w,
        p_h,
        t_x,
        t_y,
        t_w,
        t_h,
        y_is_bottom_edge=y_is_bottom_edge,
    )
    p_cx = p_x + p_w / 2.0
    p_cy = _center_y(p_y, p_h, y_is_bottom_edge=y_is_bottom_edge)
    t_cx = t_x + t_w / 2.0
    t_cy = _center_y(t_y, t_h, y_is_bottom_edge=y_is_bottom_edge)
    dx = t_cx - p_cx
    dy = t_cy - p_cy
    theta_target = jnp.arctan2(dy, dx)

    # ψ: player physical facing relative to θ_target.
    # Wrap to [-π, π] so sin/cos encoding is smooth and unambiguous.
    p_ori_rad = p_ori_deg * (jnp.pi / 180.0)
    psi = p_ori_rad - theta_target
    psi = jnp.arctan2(jnp.sin(psi), jnp.cos(psi))

    return distance, theta_target, psi, t_cx, t_cy, p_cx, p_cy, t_w, t_h


def _los_blocked_stacked(
    objects,
    target_idx,
    p_cxs,
    p_cys,
    t_cxs,
    t_cys,
    *,
    y_is_bottom_edge: bool,
) -> jnp.ndarray:
    """Per-frame LOS blocked flags (float32 [stack]) for DESTROY targets only.

    Occlusion is a shooting cue: COLLECT / AVOID / NAVIGATE keep ``los_blocked=0``
    even when objects sit on the player→target segment (ascent under sharks must
    not look "blocked" for NAVIGATE).
    """
    from concept_based_learning.environments.datatypes import LABEL_DESTROY

    raw = jax.vmap(
        lambda pcx, pcy, tcx, tcy, ox, oy, ow, oh, oact, olab: line_of_sight_blocked(
            pcx,
            pcy,
            tcx,
            tcy,
            target_idx,
            ox,
            oy,
            ow,
            oh,
            oact,
            olab,
            y_is_bottom_edge=y_is_bottom_edge,
        )
    )(
        p_cxs,
        p_cys,
        t_cxs,
        t_cys,
        objects.x,
        objects.y,
        objects.width,
        objects.height,
        objects.active,
        objects.semantic_label,
    ).astype(jnp.float32)
    # semantic_label: [stack, max_objects] → per-frame target label
    tgt_labels = objects.semantic_label[
        jnp.arange(objects.semantic_label.shape[0]), target_idx
    ]
    is_destroy = tgt_labels == LABEL_DESTROY
    return jnp.where(is_destroy, raw, jnp.float32(0.0))


# ---------------------------------------------------------------------------
# Main observation function
# ---------------------------------------------------------------------------

def _velocity_world_to_ego_legacy(vx: jnp.ndarray, vy: jnp.ndarray, p_ori_deg: jnp.ndarray):
    """Rotate world velocity into ego axes using the legacy convention."""
    speed = jnp.sqrt(vx * vx + vy * vy)
    cartesian_angle = jnp.arctan2(vy, vx)
    jaxatari_absolute_angle = cartesian_angle + (jnp.pi / 2.0)
    p_ori_rad = p_ori_deg * (jnp.pi / 180.0)
    relative_angle = jaxatari_absolute_angle - p_ori_rad
    relative_angle = jnp.arctan2(jnp.sin(relative_angle), jnp.cos(relative_angle))
    dvx_ego = speed * jnp.cos(relative_angle)
    dvy_ego = speed * jnp.sin(relative_angle)
    return dvx_ego, dvy_ego


def _single_frame_geom_legacy(
    p_x,
    p_y,
    p_w,
    p_h,
    p_ori,
    t_x,
    t_y,
    t_w,
    t_h,
    *,
    y_is_bottom_edge: bool,
):
    """Legacy per-frame geometry used by pre-refactor checkpoints."""
    p_cx = p_x + (p_w / 2.0)
    p_cy = _center_y(p_y, p_h, y_is_bottom_edge=y_is_bottom_edge)
    t_cx = t_x + (t_w / 2.0)
    t_cy = _center_y(t_y, t_h, y_is_bottom_edge=y_is_bottom_edge)

    dx = t_cx - p_cx
    dy = t_cy - p_cy
    distance = jnp.sqrt(dx ** 2 + dy ** 2)

    cartesian_angle = jnp.arctan2(dy, dx)
    jaxatari_absolute_angle = cartesian_angle + (jnp.pi / 2.0)
    p_ori_rad = p_ori * (jnp.pi / 180.0)

    relative_angle = jaxatari_absolute_angle - p_ori_rad
    relative_angle = jnp.arctan2(jnp.sin(relative_angle), jnp.cos(relative_angle))

    dx_ego = distance * jnp.cos(relative_angle)
    dy_ego = distance * jnp.sin(relative_angle)

    return (
        dx_ego,
        dy_ego,
        p_ori,
        t_w,
        t_h,
        p_w,
        p_h,
        p_cx,
        p_cy,
        t_cx,
        t_cy,
    )


def _get_ego_obs_target_aligned(raw_observation, objects, target_idx):
    """
    Build the target-aligned egocentric observation over the full JAXAtari frame stack.

    Parameters
    ----------
    raw_observation : JAXAtari raw obs PyTree; must expose raw_observation.player.{x,y,width,height,orientation}
    objects         : LabeledObjectObservation PyTree; fields x,y,width,height shaped [stack, max_objects]
    target_idx      : int32 scalar — index into the objects arrays

    Returns
    -------
    flat float32 array of shape (EGO_OBS_CHANNELS_PER_FRAME * stack_size,)
    """
    t_x_hist = objects.x[:, target_idx]
    t_y_hist = objects.y[:, target_idx]
    t_w_hist = objects.width[:, target_idx]
    t_h_hist = objects.height[:, target_idx]

    player = _get_player(raw_observation)
    y_is_bottom_edge = _uses_bottom_edge_y(raw_observation)
    (
        distances,     # nearest surface gap (edge/corner to edge/corner)
        thetas,        # centre→centre angle
        psis,
        t_cxs, t_cys,
        p_cxs, p_cys,
        t_ws,  t_hs,
    ) = jax.vmap(
        lambda px, py, pw, ph, po, tx, ty, tw, th: _frame_geometry_target_aligned(
            px,
            py,
            pw,
            ph,
            po,
            tx,
            ty,
            tw,
            th,
            y_is_bottom_edge=y_is_bottom_edge,
        )
    )(
        player.x,
        player.y,
        player.width,
        player.height,
        player.orientation,
        t_x_hist, t_y_hist, t_w_hist, t_h_hist,
    )

    # -------------------------------------------------------------------
    # Radial and tangential velocities
    # -------------------------------------------------------------------
    # World-space relative position across the stack
    rel_x = t_cxs - p_cxs
    rel_y = t_cys - p_cys

    # Finite-difference velocity; zero-pad the oldest frame (no prior data)
    d_rel_x = jnp.concatenate([jnp.zeros(1, dtype=rel_x.dtype), jnp.diff(rel_x)])
    d_rel_y = jnp.concatenate([jnp.zeros(1, dtype=rel_y.dtype), jnp.diff(rel_y)])

    # Decompose into radial (along θ_target) and tangential (perpendicular) components.
    #
    #   radial unit vector:      ( cos θ,  sin θ)
    #   tangential unit vector:  (-sin θ,  cos θ)   (90° CCW from radial)
    #
    # v_r > 0: relative position growing → target is moving away (or player lagging)
    # v_t > 0: target drifting to the right of the line-of-sight
    cos_t = jnp.cos(thetas)
    sin_t = jnp.sin(thetas)
    v_r =  d_rel_x * cos_t + d_rel_y * sin_t
    v_t = -d_rel_x * sin_t + d_rel_y * cos_t

    # -------------------------------------------------------------------
    # Stack and normalise
    # -------------------------------------------------------------------
    if _USE_EGO_SIZE_FEATURES:
        size_w = t_ws / SCREEN_WIDTH
        size_h = t_hs / SCREEN_HEIGHT
    else:
        size_w = jnp.zeros_like(t_ws, dtype=jnp.float32)
        size_h = jnp.zeros_like(t_hs, dtype=jnp.float32)

    if _USE_EGO_VELOCITY_FEATURES:
        v_r_norm = v_r / MAX_SPEED
        v_t_norm = v_t / MAX_SPEED
    else:
        v_r_norm = jnp.zeros_like(v_r, dtype=jnp.float32)
        v_t_norm = jnp.zeros_like(v_t, dtype=jnp.float32)

    channels = distance_features_from_pixels(distances)
    if _USE_EGO_VELOCITY_FEATURES:
        channels.extend([v_r_norm, v_t_norm])
    if _USE_EGO_SIZE_FEATURES:
        channels.extend([size_w, size_h])
    if _EGO_ANGLE_BINS:
        if _EGO_ANGLE_SINCOS_SNAP:
            sin_t, cos_t = quantize_angle_sincos(thetas, _EGO_ANGLE_BINS)
            channels.extend([sin_t, cos_t])
        else:
            bin_feats = triangular_angle_bins(thetas, _EGO_ANGLE_BINS)
            if _EGO_ANGLE_BINS_SCALAR:
                n = jnp.float32(bin_feats.shape[-1] - 1)
                idx = jnp.argmax(bin_feats, axis=-1).astype(jnp.float32)
                scalar = jnp.where(n > 0, idx / n, jnp.zeros_like(idx))
                channels.append(scalar)
            else:
                channels.extend([bin_feats[:, i] for i in range(int(_EGO_ANGLE_BINS))])
    else:
        channels.extend([jnp.sin(psis), jnp.cos(psis)])
    if _USE_EGO_BLOCKING_FEATURES:
        channels.append(
            _los_blocked_stacked(
                objects,
                target_idx,
                p_cxs,
                p_cys,
                t_cxs,
                t_cys,
                y_is_bottom_edge=y_is_bottom_edge,
            )
        )
    skill_extras = build_skill_ego_extras_stacked(
        raw_observation, objects=objects, target_idx=target_idx
    )
    if skill_extras.shape[-1] > 0:
        stacked = jnp.concatenate([jnp.stack(channels, axis=-1), skill_extras], axis=-1)
    else:
        stacked = jnp.stack(channels, axis=-1)
    return stacked.astype(jnp.float32).flatten()


def _get_ego_obs_pov_relative(raw_observation, objects, target_idx, heading_deg=None):
    """POV-relative ego obs: view = toward target; ψ = facing − view; sin/cos(ψ).

    When velocity is enabled, emit one signed **approach** channel in the view
    frame (line of sight to target): positive = target closing on the player,
    negative = receding. Lateral drift is omitted — view is locked on target.

    ``heading_deg`` overrides the sprite orientation as the facing used for ψ,
    which is how the POV turn primitives close their control loop. It is a single
    scalar broadcast across the frame stack: the integrator only tracks the
    current heading, so ψ history varies through the target bearing alone.
    """
    from concept_based_learning.observations.pov_relative import pov_facing_psi_rad

    t_x_hist = objects.x[:, target_idx]
    t_y_hist = objects.y[:, target_idx]
    t_w_hist = objects.width[:, target_idx]
    t_h_hist = objects.height[:, target_idx]

    player = _get_player(raw_observation)
    y_is_bottom_edge = _uses_bottom_edge_y(raw_observation)
    (
        distances,
        thetas,
        _psis_unused,
        t_cxs,
        t_cys,
        p_cxs,
        p_cys,
        t_ws,
        t_hs,
    ) = jax.vmap(
        lambda px, py, pw, ph, po, tx, ty, tw, th: _frame_geometry_target_aligned(
            px,
            py,
            pw,
            ph,
            po,
            tx,
            ty,
            tw,
            th,
            y_is_bottom_edge=y_is_bottom_edge,
        )
    )(
        player.x,
        player.y,
        player.width,
        player.height,
        player.orientation,
        t_x_hist,
        t_y_hist,
        t_w_hist,
        t_h_hist,
    )

    if heading_deg is None:
        facing_deg = player.orientation
    else:
        facing_deg = jnp.broadcast_to(
            jnp.asarray(heading_deg, dtype=jnp.float32), thetas.shape
        )
    psis = pov_facing_psi_rad(facing_deg, thetas)

    rel_x = t_cxs - p_cxs
    rel_y = t_cys - p_cys
    d_rel_x = jnp.concatenate([jnp.zeros(1, dtype=rel_x.dtype), jnp.diff(rel_x)])
    d_rel_y = jnp.concatenate([jnp.zeros(1, dtype=rel_y.dtype), jnp.diff(rel_y)])

    # View-radial speed along θ_view (bearing to target). Standard v_r > 0 means
    # separation growing (receding); negate so positive = closing on the player.
    cos_t = jnp.cos(thetas)
    sin_t = jnp.sin(thetas)
    v_r = d_rel_x * cos_t + d_rel_y * sin_t
    v_approach = -v_r

    if _USE_EGO_SIZE_FEATURES:
        size_w = t_ws / SCREEN_WIDTH
        size_h = t_hs / SCREEN_HEIGHT
    else:
        size_w = jnp.zeros_like(t_ws, dtype=jnp.float32)
        size_h = jnp.zeros_like(t_hs, dtype=jnp.float32)

    channels = distance_features_from_pixels(distances)
    if _USE_EGO_VELOCITY_FEATURES:
        channels.append(v_approach / MAX_SPEED)
    if _USE_EGO_SIZE_FEATURES:
        channels.extend([size_w, size_h])
    if _EGO_PSI_ANGLE_BINS:
        bin_feats = triangular_angle_bins(psis, _EGO_PSI_ANGLE_BINS)
        channels.extend([bin_feats[:, i] for i in range(int(_EGO_PSI_ANGLE_BINS))])
    else:
        channels.extend([jnp.sin(psis), jnp.cos(psis)])
    if _USE_EGO_BLOCKING_FEATURES:
        channels.append(
            _los_blocked_stacked(
                objects,
                target_idx,
                p_cxs,
                p_cys,
                t_cxs,
                t_cys,
                y_is_bottom_edge=y_is_bottom_edge,
            )
        )
    skill_extras = build_skill_ego_extras_stacked(
        raw_observation, objects=objects, target_idx=target_idx
    )
    if skill_extras.shape[-1] > 0:
        stacked = jnp.concatenate([jnp.stack(channels, axis=-1), skill_extras], axis=-1)
    else:
        stacked = jnp.stack(channels, axis=-1)
    return stacked.astype(jnp.float32).flatten()


def _frame_geometry_fixed_bilateral(
    p_x,
    p_y,
    p_w,
    p_h,
    t_x,
    t_y,
    t_w,
    t_h,
    *,
    y_is_bottom_edge: bool,
    axis: str,
):
    """Per-frame distance + forward-hemisphere φ under virtual bilateral sectors."""
    from concept_based_learning.observations.pov_fixed_bilateral import (
        forward_hemisphere_phi_rad,
        target_sector_idx,
    )

    distance = aabb_surface_gap_px(
        p_x,
        p_y,
        p_w,
        p_h,
        t_x,
        t_y,
        t_w,
        t_h,
        y_is_bottom_edge=y_is_bottom_edge,
    )
    p_cx = p_x + p_w / 2.0
    p_cy = _center_y(p_y, p_h, y_is_bottom_edge=y_is_bottom_edge)
    t_cx = t_x + t_w / 2.0
    t_cy = _center_y(t_y, t_h, y_is_bottom_edge=y_is_bottom_edge)
    dx = t_cx - p_cx
    dy = t_cy - p_cy
    sector = target_sector_idx(dx, dy, axis=axis)
    phi = forward_hemisphere_phi_rad(dx, dy, sector, axis=axis)
    return distance, phi, t_cx, t_cy, p_cx, p_cy, t_w, t_h


def _get_ego_obs_pov_fixed_bilateral(raw_observation, objects, target_idx):
    """Virtual bilateral sectors: distance + forward-hemisphere φ (no velocity)."""
    from concept_based_learning.observations.pov_fixed_bilateral import (
        active_bilateral_axis,
        forward_hemisphere_angle_bins,
    )

    axis = active_bilateral_axis()

    t_x_hist = objects.x[:, target_idx]
    t_y_hist = objects.y[:, target_idx]
    t_w_hist = objects.width[:, target_idx]
    t_h_hist = objects.height[:, target_idx]

    player = _get_player(raw_observation)
    y_is_bottom_edge = _uses_bottom_edge_y(raw_observation)
    distances, phis, t_cxs, t_cys, p_cxs, p_cys, t_ws, t_hs = jax.vmap(
        lambda px, py, pw, ph, tx, ty, tw, th: _frame_geometry_fixed_bilateral(
            px,
            py,
            pw,
            ph,
            tx,
            ty,
            tw,
            th,
            y_is_bottom_edge=y_is_bottom_edge,
            axis=axis,
        )
    )(
        player.x,
        player.y,
        player.width,
        player.height,
        t_x_hist,
        t_y_hist,
        t_w_hist,
        t_h_hist,
    )

    if _USE_EGO_SIZE_FEATURES:
        size_w = t_ws / SCREEN_WIDTH
        size_h = t_hs / SCREEN_HEIGHT
    else:
        size_w = jnp.zeros_like(t_ws, dtype=jnp.float32)
        size_h = jnp.zeros_like(t_hs, dtype=jnp.float32)

    channels = distance_features_from_pixels(distances)
    if _USE_EGO_SIZE_FEATURES:
        channels.extend([size_w, size_h])
    if _EGO_PSI_ANGLE_BINS:
        bin_feats = forward_hemisphere_angle_bins(phis, _EGO_PSI_ANGLE_BINS)
        channels.extend([bin_feats[:, i] for i in range(int(_EGO_PSI_ANGLE_BINS))])
    else:
        channels.extend([jnp.sin(phis), jnp.cos(phis)])
    if _USE_EGO_BLOCKING_FEATURES:
        channels.append(
            _los_blocked_stacked(
                objects,
                target_idx,
                p_cxs,
                p_cys,
                t_cxs,
                t_cys,
                y_is_bottom_edge=y_is_bottom_edge,
            )
        )
    skill_extras = build_skill_ego_extras_stacked(
        raw_observation, objects=objects, target_idx=target_idx
    )
    if skill_extras.shape[-1] > 0:
        stacked = jnp.concatenate([jnp.stack(channels, axis=-1), skill_extras], axis=-1)
    else:
        stacked = jnp.stack(channels, axis=-1)
    return stacked.astype(jnp.float32).flatten()


def _get_ego_obs_legacy(raw_observation, objects, target_idx):
    """Legacy 10-channel egocentric observations from pre-refactor training."""
    t_x_history = objects.x[:, target_idx]
    t_y_history = objects.y[:, target_idx]
    t_w_history = objects.width[:, target_idx]
    t_h_history = objects.height[:, target_idx]

    player = _get_player(raw_observation)
    y_is_bottom_edge = _uses_bottom_edge_y(raw_observation)
    (
        dx_ego,
        dy_ego,
        p_ori_deg,
        t_w,
        t_h,
        p_w,
        p_h,
        p_cx,
        p_cy,
        t_cx,
        t_cy,
    ) = jax.vmap(
        lambda px, py, pw, ph, po, tx, ty, tw, th: _single_frame_geom_legacy(
            px,
            py,
            pw,
            ph,
            po,
            tx,
            ty,
            tw,
            th,
            y_is_bottom_edge=y_is_bottom_edge,
        )
    )(
        player.x,
        player.y,
        player.width,
        player.height,
        player.orientation,
        t_x_history,
        t_y_history,
        t_w_history,
        t_h_history,
    )

    rel_x = t_cx - p_cx
    rel_y = t_cy - p_cy
    dvx_world = jnp.concatenate([jnp.zeros(1, dtype=rel_x.dtype), jnp.diff(rel_x)])
    dvy_world = jnp.concatenate([jnp.zeros(1, dtype=rel_y.dtype), jnp.diff(rel_y)])

    dvx_ego, dvy_ego = jax.vmap(_velocity_world_to_ego_legacy)(
        dvx_world, dvy_world, p_ori_deg
    )

    p_ori_rad = p_ori_deg * (jnp.pi / 180.0)
    sin_o = jnp.sin(p_ori_rad)
    cos_o = jnp.cos(p_ori_rad)

    if _USE_EGO_SIZE_FEATURES:
        t_w_norm = t_w / SCREEN_WIDTH
        t_h_norm = t_h / SCREEN_HEIGHT
        p_w_norm = p_w / SCREEN_WIDTH
        p_h_norm = p_h / SCREEN_HEIGHT
    else:
        t_w_norm = jnp.zeros_like(t_w, dtype=jnp.float32)
        t_h_norm = jnp.zeros_like(t_h, dtype=jnp.float32)
        p_w_norm = jnp.zeros_like(p_w, dtype=jnp.float32)
        p_h_norm = jnp.zeros_like(p_h, dtype=jnp.float32)

    if _USE_EGO_VELOCITY_FEATURES:
        dvx_norm = dvx_ego / MAX_SPEED
        dvy_norm = dvy_ego / MAX_SPEED
    else:
        dvx_norm = jnp.zeros_like(dvx_ego, dtype=jnp.float32)
        dvy_norm = jnp.zeros_like(dvy_ego, dtype=jnp.float32)

    channels = [
        dx_ego / SCREEN_WIDTH,
        dy_ego / SCREEN_HEIGHT,
    ]
    if _USE_EGO_VELOCITY_FEATURES:
        channels.extend([dvx_norm, dvy_norm])
    if _USE_EGO_SIZE_FEATURES:
        channels.extend([t_w_norm, t_h_norm, p_w_norm, p_h_norm])
    channels.extend([sin_o, cos_o])
    if _USE_EGO_BLOCKING_FEATURES:
        channels.append(
            _los_blocked_stacked(
                objects,
                target_idx,
                p_cx,
                p_cy,
                t_cx,
                t_cy,
                y_is_bottom_edge=y_is_bottom_edge,
            )
        )
    skill_extras = build_skill_ego_extras_stacked(
        raw_observation, objects=objects, target_idx=target_idx
    )
    if skill_extras.shape[-1] > 0:
        stacked_ego = jnp.concatenate([jnp.stack(channels, axis=-1), skill_extras], axis=-1)
    else:
        stacked_ego = jnp.stack(channels, axis=-1)
    return stacked_ego.astype(jnp.float32).flatten()


def get_ego_obs(raw_observation, objects, target_idx, heading_deg=None):
    """
    Build egocentric observations using the currently selected format.

    Toggle via `set_ego_obs_format(use_legacy=True/False)`.

    ``heading_deg`` is only meaningful under the POV mapper with turn primitives,
    where ψ must be measured against the integrated heading rather than the
    sprite orientation. Omitting it there would silently feed the network a
    facing it never commanded, so that case raises instead.
    """
    from concept_based_learning.observations.pov_relative import using_pov_relative
    from concept_based_learning.observations.pov_fixed_bilateral import (
        using_pov_fixed_bilateral,
    )
    from concept_based_learning.observations.pov_turn import using_pov_turn_primitives

    if using_pov_fixed_bilateral():
        return _get_ego_obs_pov_fixed_bilateral(raw_observation, objects, target_idx)
    if using_pov_relative():
        if heading_deg is None and using_pov_turn_primitives():
            raise ValueError(
                "USE_POV_TURN_PRIMITIVES is on but get_ego_obs was called without "
                "heading_deg. Pass PolicyState.heading_deg from the rollout carry."
            )
        return _get_ego_obs_pov_relative(
            raw_observation, objects, target_idx, heading_deg
        )
    if _USE_LEGACY_EGO_OBS:
        return _get_ego_obs_legacy(raw_observation, objects, target_idx)
    return _get_ego_obs_target_aligned(raw_observation, objects, target_idx)
