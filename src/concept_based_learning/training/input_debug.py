"""Optional rollout/training input checks for masked input-mode bugs."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def input_debug_enabled(config) -> bool:
    return bool(config.get("DEBUG_INPUTS", False))


def input_debug_assert(config) -> bool:
    if not input_debug_enabled(config):
        return False
    return bool(config.get("DEBUG_INPUTS_ASSERT", True))


def _log_every_n_updates(config) -> int:
    return max(1, int(config.get("DEBUG_INPUTS_LOG_EVERY_N_UPDATES", 1)))


def _host_tensor_summary(name: str, arr: Any) -> str:
    if arr is None:
        return f"{name}: None"
    try:
        a = np.asarray(arr)
    except (TypeError, ValueError):
        return f"{name}: <non-array {type(arr).__name__}>"
    if a.dtype == object or not np.issubdtype(a.dtype, np.number):
        return f"{name}: shape={tuple(a.shape)} dtype={a.dtype} (non-numeric; skipped stats)"
    finite = bool(np.isfinite(a).all())
    return (
        f"{name}: shape={tuple(a.shape)} dtype={a.dtype} finite={finite} "
        f"min={float(np.nanmin(a)):.4g} max={float(np.nanmax(a)):.4g} "
        f"mean={float(np.nanmean(a)):.4g} std={float(np.nanstd(a)):.4g}"
    )


def _host_assert_finite(name: str, arr) -> None:
    a = np.asarray(arr)
    if a.dtype == object or not np.issubdtype(a.dtype, np.number):
        return
    if not np.isfinite(a).all():
        raise ValueError(f"DEBUG_INPUTS: {name} contains non-finite values.")


def _host_assert_rank(arr, expected: int, name: str) -> None:
    a = np.asarray(arr)
    if a.ndim != expected:
        raise ValueError(
            f"DEBUG_INPUTS: {name} expected rank {expected}, got {a.ndim}."
        )


def _host_assert_rank_at_least(arr, min_rank: int, name: str) -> None:
    a = np.asarray(arr)
    if a.ndim < min_rank:
        raise ValueError(
            f"DEBUG_INPUTS: {name} expected rank >= {min_rank}, got {a.ndim}."
        )


def _host_assert_shape_prefix_equal(a, b, n: int, msg: str) -> None:
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    if tuple(a_np.shape[:n]) != tuple(b_np.shape[:n]):
        raise ValueError(f"DEBUG_INPUTS: {msg} ({a_np.shape} vs {b_np.shape}).")


def _host_log(site: str, pairs: Sequence[tuple[str, Any]]) -> None:
    lines = [f"[DEBUG_INPUTS] {site}"]
    for name, arr in pairs:
        if arr is None:
            lines.append(f"  {name}: None")
        else:
            lines.append(f"  {_host_tensor_summary(name, arr)}")
    print("\n".join(lines), flush=True)


def host_check_probe_inputs(
    config,
    *,
    probe_ego,
    probe_pixel,
    probe_target,
    meta_pixel=None,
) -> None:
    """Run on host (e.g. via jax.debug.callback) with concrete probe arrays."""
    if not input_debug_enabled(config):
        return

    skill_mode = str(config.get("SKILL_INPUT_MODE", "ego")).lower().strip()
    meta_mode = str(config.get("META_INPUT_MODE", "flat")).lower().strip()
    max_objects = int(config["MAX_OBJECTS"])

    _host_log(
        "probe/init",
        [
            ("probe_ego", probe_ego),
            ("probe_pixel", probe_pixel),
            ("probe_target", probe_target),
            ("meta_pixel", meta_pixel),
        ],
    )

    if input_debug_assert(config):
        _host_assert_finite("probe_ego", probe_ego)
        if probe_pixel is not None and skill_mode in ("pixel_embed", "pixel_embed_cond"):
            _host_assert_finite("probe_pixel", probe_pixel)

    tgt = int(np.asarray(probe_target).reshape(()))
    if not (0 <= tgt < max_objects):
        raise ValueError(
            f"DEBUG_INPUTS: probe_target={tgt} out of range [0, {max_objects})."
        )

    if skill_mode in ("pixel_embed", "pixel_embed_cond"):
        if probe_pixel is None:
            raise ValueError(
                f"DEBUG_INPUTS: SKILL_INPUT_MODE={skill_mode} but probe pixel_obs is None."
            )
        if int(np.asarray(probe_pixel).ndim) < 4:
            raise ValueError(
                "DEBUG_INPUTS: probe pixel_obs must have rank >= 4 "
                f"[..., T, H, W, C], got shape {np.asarray(probe_pixel).shape}."
            )
    if meta_mode == "pixel_embed":
        if meta_pixel is None:
            raise ValueError(
                "DEBUG_INPUTS: META_INPUT_MODE=pixel_embed but probe pixel_observation is None."
            )
        if int(np.asarray(meta_pixel).ndim) < 4:
            raise ValueError(
                "DEBUG_INPUTS: probe meta pixel_observation must have rank >= 4, "
                f"got shape {np.asarray(meta_pixel).shape}."
            )

    if skill_mode == "ego":
        ego_dim = int(np.asarray(probe_ego).shape[-1])
        if ego_dim > 64:
            print(
                f"[DEBUG_INPUTS] warning: SKILL_INPUT_MODE=ego but probe_ego dim={ego_dim} "
                "(expected small egocentric feature vector).",
                flush=True,
            )
    if skill_mode == "oc_embed":
        ego_dim = int(np.asarray(probe_ego).shape[-1])
        if ego_dim < 64:
            print(
                f"[DEBUG_INPUTS] warning: SKILL_INPUT_MODE=oc_embed but probe_ego dim={ego_dim} "
                "(expected flattened object-centric vector).",
                flush=True,
            )


def maybe_jax_check_probe_inputs(
    config,
    *,
    probe_ego,
    probe_pixel,
    probe_target,
    meta_pixel=None,
) -> None:
    """Runtime probe checks safe inside jitted train()."""
    if not input_debug_enabled(config):
        return

    def _cb(ego, pixel, target, meta_pix):
        host_check_probe_inputs(
            config,
            probe_ego=ego,
            probe_pixel=None if int(np.asarray(pixel).size) == 0 else pixel,
            probe_target=target,
            meta_pixel=None if int(np.asarray(meta_pix).size) == 0 else meta_pix,
        )

    skill_mode = str(config.get("SKILL_INPUT_MODE", "ego")).lower().strip()
    meta_mode = str(config.get("META_INPUT_MODE", "flat")).lower().strip()

    pixel_arg = (
        probe_pixel
        if skill_mode in ("pixel_embed", "pixel_embed_cond") and probe_pixel is not None
        else jnp.array([], dtype=jnp.float32)
    )
    meta_arg = (
        meta_pixel
        if meta_mode == "pixel_embed" and meta_pixel is not None
        else jnp.array([], dtype=jnp.float32)
    )
    jax.debug.callback(_cb, probe_ego, pixel_arg, probe_target, meta_arg)


def _host_assert_rollout_transitions(
    ego_obs,
    next_ego_obs,
    target_idx,
    pixel_obs,
    next_pixel_obs,
    ego_obs_all,
    next_ego_obs_all,
    relabel_mask,
    *,
    max_objects: int,
    no_target_idx: int,
    use_pixel_skill: bool,
    use_pixel_meta: bool,
    relabel_all_objects: bool,
) -> None:
    _host_assert_finite("ego_obs", ego_obs)
    _host_assert_finite("next_ego_obs", next_ego_obs)
    _host_assert_rank(ego_obs, 3, "ego_obs")
    _host_assert_rank(next_ego_obs, 3, "next_ego_obs")
    _host_assert_rank(target_idx, 2, "target_idx")
    if np.asarray(ego_obs).shape != np.asarray(next_ego_obs).shape:
        raise ValueError("DEBUG_INPUTS: ego_obs and next_ego_obs shape mismatch.")

    tgt = np.asarray(target_idx)
    if np.any(tgt < 0) or np.any(tgt >= max_objects):
        raise ValueError("DEBUG_INPUTS: target_idx out of range.")

    if relabel_all_objects:
        _host_assert_rank(ego_obs_all, 4, "ego_obs_all")
        _host_assert_rank(next_ego_obs_all, 4, "next_ego_obs_all")
        _host_assert_shape_prefix_equal(
            ego_obs_all, next_ego_obs_all, 3, "ego_obs_all/next_ego_obs_all lead dims mismatch"
        )
        _host_assert_shape_prefix_equal(
            ego_obs_all, ego_obs, 2, "ego_obs_all/ego_obs T/E dims mismatch"
        )

    if use_pixel_skill:
        _host_assert_rank_at_least(pixel_obs, 5, "pixel_obs")
        _host_assert_rank_at_least(next_pixel_obs, 5, "next_pixel_obs")
        _host_assert_finite("pixel_obs", pixel_obs)
        _host_assert_finite("next_pixel_obs", next_pixel_obs)
        _host_assert_shape_prefix_equal(
            pixel_obs, ego_obs, 2, "pixel_obs/ego_obs T/E dims mismatch"
        )
        _host_assert_shape_prefix_equal(
            next_pixel_obs, next_ego_obs, 2, "next_pixel_obs/next_ego_obs T/E dims mismatch"
        )

    if use_pixel_meta:
        _host_assert_rank_at_least(pixel_obs, 5, "meta pixel_obs")

    no_target = tgt == no_target_idx
    if np.any(no_target):
        masked = np.where(no_target, np.asarray(relabel_mask), 0.0)
        if not np.all(masked == 0.0):
            raise ValueError("DEBUG_INPUTS: no_target slots must not be relabeled.")


def _host_assert_skill_minibatch(
    ego_obs,
    target_idx,
    pixel_obs,
    *,
    max_objects: int,
    use_pixel_skill: bool,
) -> None:
    _host_assert_finite("ego_obs", ego_obs)
    _host_assert_rank_at_least(ego_obs, 2, "ego_obs")
    tgt = np.asarray(target_idx)
    if np.any(tgt < 0) or np.any(tgt >= max_objects):
        raise ValueError("DEBUG_INPUTS: minibatch target_idx out of range.")
    if use_pixel_skill:
        _host_assert_rank_at_least(pixel_obs, 4, "pixel_obs")
        _host_assert_finite("pixel_obs", pixel_obs)
        ego_np = np.asarray(ego_obs)
        pix_np = np.asarray(pixel_obs)
        ego_lead = ego_np.shape[:-1]
        pix_lead = pix_np.shape[:-4]
        if len(pix_lead) > len(ego_lead) or tuple(ego_lead[: len(pix_lead)]) != tuple(
            pix_lead
        ):
            raise ValueError(
                "DEBUG_INPUTS: pixel_obs shared lead dims must prefix ego_obs/object axes."
            )


def _jax_host_log_rollout(
    site: str,
    update_idx: int,
    ego_obs_shape,
    next_ego_obs_shape,
    target_min: int,
    target_max: int,
    pixel_shape,
    next_pixel_shape,
    relabel_mask_mean: float,
    ego_obs_all_shape,
    next_ego_obs_all_shape,
) -> None:
    _host_log(
        f"{site} (update={update_idx})",
        [
            ("ego_obs.shape", np.array(ego_obs_shape, dtype=np.int32)),
            ("next_ego_obs.shape", np.array(next_ego_obs_shape, dtype=np.int32)),
            ("target_idx.min", np.array(target_min)),
            ("target_idx.max", np.array(target_max)),
            ("pixel_obs.shape", np.array(pixel_shape, dtype=np.int32) if pixel_shape is not None else None),
            ("next_pixel_obs.shape", np.array(next_pixel_shape, dtype=np.int32) if next_pixel_shape is not None else None),
            ("relabel_mask.mean", np.array(relabel_mask_mean)),
            ("ego_obs_all.shape", np.array(ego_obs_all_shape, dtype=np.int32) if ego_obs_all_shape is not None else None),
            ("next_ego_obs_all.shape", np.array(next_ego_obs_all_shape, dtype=np.int32) if next_ego_obs_all_shape is not None else None),
        ],
    )


def maybe_jax_check_rollout_transitions(
    config,
    *,
    transitions,
    update_idx: jnp.ndarray,
    max_objects: int,
    no_target_idx: int,
    skill_input_mode: str,
    meta_input_mode: str,
    relabel_all_objects: bool,
) -> None:
    """JAX-safe rollout buffer checks; logs on host via callback."""
    if not input_debug_enabled(config):
        return

    skill_mode = skill_input_mode.lower().strip()
    meta_mode = meta_input_mode.lower().strip()
    use_pixel_skill = skill_mode in ("pixel_embed", "pixel_embed_cond")
    use_pixel_meta = meta_mode == "pixel_embed"

    if input_debug_assert(config):
        def _assert_cb(
            ego_obs,
            next_ego_obs,
            target_idx,
            pixel_obs,
            next_pixel_obs,
            ego_obs_all,
            next_ego_obs_all,
            relabel_mask,
        ):
            _host_assert_rollout_transitions(
                ego_obs,
                next_ego_obs,
                target_idx,
                pixel_obs,
                next_pixel_obs,
                ego_obs_all,
                next_ego_obs_all,
                relabel_mask,
                max_objects=max_objects,
                no_target_idx=no_target_idx,
                use_pixel_skill=use_pixel_skill,
                use_pixel_meta=use_pixel_meta,
                relabel_all_objects=relabel_all_objects,
            )

        jax.debug.callback(
            _assert_cb,
            transitions.ego_obs,
            transitions.next_ego_obs,
            transitions.target_idx,
            transitions.pixel_obs,
            transitions.next_pixel_obs,
            transitions.ego_obs_all,
            transitions.next_ego_obs_all,
            transitions.relabel_mask,
        )

    every = _log_every_n_updates(config)
    should_log = (update_idx % every) == 0

    def _do_log(_):
        def _cb(
            update_i,
            ego_shape,
            next_ego_shape,
            target_min,
            target_max,
            relabel_mean,
            pixel_shape,
            next_pixel_shape,
            ego_all_shape,
            next_ego_all_shape,
        ):
            _jax_host_log_rollout(
                "rollout/transition",
                int(np.asarray(update_i)),
                tuple(int(x) for x in np.asarray(ego_shape)),
                tuple(int(x) for x in np.asarray(next_ego_shape)),
                int(np.asarray(target_min)),
                int(np.asarray(target_max)),
                tuple(int(x) for x in np.asarray(pixel_shape))
                if int(np.asarray(pixel_shape)[0]) >= 0
                else None,
                tuple(int(x) for x in np.asarray(next_pixel_shape))
                if int(np.asarray(next_pixel_shape)[0]) >= 0
                else None,
                float(np.asarray(relabel_mean)),
                tuple(int(x) for x in np.asarray(ego_all_shape))
                if int(np.asarray(ego_all_shape)[0]) >= 0
                else None,
                tuple(int(x) for x in np.asarray(next_ego_all_shape))
                if int(np.asarray(next_ego_all_shape)[0]) >= 0
                else None,
            )

        jax.debug.callback(
            _cb,
            update_idx,
            jnp.array(jnp.shape(transitions.ego_obs), dtype=jnp.int32),
            jnp.array(jnp.shape(transitions.next_ego_obs), dtype=jnp.int32),
            jnp.min(transitions.target_idx),
            jnp.max(transitions.target_idx),
            jnp.mean(transitions.relabel_mask.astype(jnp.float32)),
            jnp.array(jnp.shape(transitions.pixel_obs), dtype=jnp.int32)
            if use_pixel_skill
            else jnp.array([-1], dtype=jnp.int32),
            jnp.array(jnp.shape(transitions.next_pixel_obs), dtype=jnp.int32)
            if use_pixel_skill
            else jnp.array([-1], dtype=jnp.int32),
            jnp.array(jnp.shape(transitions.ego_obs_all), dtype=jnp.int32)
            if relabel_all_objects
            else jnp.array([-1], dtype=jnp.int32),
            jnp.array(jnp.shape(transitions.next_ego_obs_all), dtype=jnp.int32)
            if relabel_all_objects
            else jnp.array([-1], dtype=jnp.int32),
        )
        return 0

    jax.lax.cond(should_log, _do_log, lambda _: 0, operand=0)


def maybe_jax_check_skill_minibatch(
    config,
    *,
    update_idx: jnp.ndarray,
    ego_obs,
    target_idx,
    pixel_obs,
    use_pixel_skill: bool,
    site: str,
) -> None:
    if not input_debug_enabled(config):
        return

    max_objects = int(config["MAX_OBJECTS"])

    if input_debug_assert(config):
        def _assert_cb(e_obs, t_idx, p_obs):
            _host_assert_skill_minibatch(
                e_obs,
                t_idx,
                p_obs if use_pixel_skill else np.zeros((0,), dtype=np.float32),
                max_objects=max_objects,
                use_pixel_skill=use_pixel_skill,
            )

        pix_arg = (
            pixel_obs
            if use_pixel_skill and pixel_obs is not None
            else jnp.zeros((0,), dtype=jnp.float32)
        )
        jax.debug.callback(_assert_cb, ego_obs, target_idx, pix_arg)

    every = _log_every_n_updates(config)
    should_log = (update_idx % every) == 0
    site_label = site

    def _do_log(_):
        def _cb(update_i, ego_shape, target_min, target_max, pixel_shape):
            pix = None
            if use_pixel_skill and int(np.asarray(pixel_shape)[0]) >= 0:
                pix = np.asarray(pixel_shape, dtype=np.int32)
            _host_log(
                f"{site_label} (update={int(np.asarray(update_i))})",
                [
                    ("ego_obs.shape", np.asarray(ego_shape, dtype=np.int32)),
                    ("target_idx.min", np.asarray(target_min)),
                    ("target_idx.max", np.asarray(target_max)),
                    ("pixel_obs.shape", pix),
                ],
            )

        pix_shape = (
            jnp.array(jnp.shape(pixel_obs), dtype=jnp.int32)
            if use_pixel_skill and pixel_obs is not None
            else jnp.array([-1], dtype=jnp.int32)
        )
        jax.debug.callback(
            _cb,
            update_idx,
            jnp.array(jnp.shape(ego_obs), dtype=jnp.int32),
            jnp.min(target_idx),
            jnp.max(target_idx),
            pix_shape,
        )
        return 0

    jax.lax.cond(should_log, _do_log, lambda _: 0, operand=0)
