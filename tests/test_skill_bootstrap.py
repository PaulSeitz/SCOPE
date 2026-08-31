"""Chunked / sequential pixel-skill CNN helpers."""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.training.skill_bootstrap import (
    PIXEL_CNN_TARGET_FRAMES,
    map_over_skills,
    resolve_skill_chunk_steps,
    skill_network_q_max,
)


def test_resolve_pixel_default_caps_frames():
    steps = resolve_skill_chunk_steps(
        num_steps=128, num_envs=128, configured=0, use_pixel=True
    )
    assert steps == PIXEL_CNN_TARGET_FRAMES // 128
    assert steps * 128 <= PIXEL_CNN_TARGET_FRAMES


def test_resolve_ego_keeps_full_horizon():
    steps = resolve_skill_chunk_steps(
        num_steps=128, num_envs=128, configured=0, use_pixel=False
    )
    assert steps == 128


def test_resolve_explicit_chunk_wins():
    steps = resolve_skill_chunk_steps(
        num_steps=128, num_envs=128, configured=16, use_pixel=True
    )
    assert steps == 16


def test_map_over_skills_scan_matches_vmap():
    def fn(w, i):
        return w * (i + 1)

    ws = jnp.array([1.0, 2.0, 3.0, 4.0])
    idxs = jnp.arange(4)
    vmapped = map_over_skills(fn, ws, idxs, sequential=False)
    scanned = map_over_skills(fn, ws, idxs, sequential=True)
    assert jnp.allclose(vmapped, scanned)


def test_skill_network_q_max_chunk_matches_full():
    def apply_fn(vars_, x, train, *, target_idx, pixel_obs, objects):
        del vars_, train, target_idx, pixel_obs, objects
        # Fake Q: [T, E, A]
        return jnp.stack([x, x + 1.0], axis=-1)

    ego = jnp.arange(12, dtype=jnp.float32).reshape(6, 2)
    tgt = jnp.zeros((6, 2), dtype=jnp.int32)
    params = {}
    stats = {}
    full = skill_network_q_max(
        apply_fn,
        params,
        stats,
        ego,
        target_idx=tgt,
        pixel_obs=None,
        chunk_steps=6,
        chunk=False,
    )
    chunked = skill_network_q_max(
        apply_fn,
        params,
        stats,
        ego,
        target_idx=tgt,
        pixel_obs=None,
        chunk_steps=2,
        chunk=True,
    )
    assert full.shape == (6, 2)
    assert jnp.allclose(full, chunked)
    assert jnp.allclose(full, ego + 1.0)
