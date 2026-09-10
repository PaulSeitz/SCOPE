#!/usr/bin/env python3
"""Diagnose Seaquest→Pong honest transfer: ego histogram, legal mask, φ/lat motion.

Runs a short Python greedy loop (not the jitted eval harness) so we can log
per-step ego actions, legal masks, body-frame (fwd, lat, φ), and paddle Δy.

Example:
  uv run python scripts/diagnose_pong_transfer.py \\
    --ckpt outputs/models/paper_eval/rq1/seaquest_skills_ego_pov_clip50_facing_psibins8_v2/seaquest/hierarchical_pqn_seaquest_seed44.safetensors \\
    --eval-yaml configs/experiment/paper_eval/rq2/pong_transfer_ego_pov_clip50_facing_psibins8_v2.yaml \\
    --episodes 2 --max-frames 4000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from concept_based_learning.environments.env_manager import EnvManager  # noqa: E402
from concept_based_learning.environments.heuristics import (  # noqa: E402
    resolve_heuristic_target_fn,
)
from concept_based_learning.observations.action_space import (  # noqa: E402
    fire_ego_action_indices,
    make_realized_ego_mask_fn,
    make_legal_ego_mask_fn,
    legal_ego_mask_mode_from_config,
    mask_q_with_legal_ego,
)
from concept_based_learning.observations.egocentric import (  # noqa: E402
    configure_ego_obs_from_config,
    get_ego_obs,
)
from concept_based_learning.observations.pov_facing_frame import (  # noqa: E402
    facing_frame_phi_rad,
    resolve_facing_deg,
    world_delta_to_facing_frame,
)
from concept_based_learning.paper_eval.config_merge import (  # noqa: E402
    flatten_experiment_cfg,
    merge_paper_eval_yaml,
)
from concept_based_learning.paper_eval.bearing_conditioning import (  # noqa: E402
    ABS_LAT_BUCKETS,
    SIGNED_LAT_BUCKETS,
    mean_pairwise_tv,
    signed_lat_bucket,
)
from concept_based_learning.networks.skill_network import q_network_from_config  # noqa: E402
from concept_based_learning.observations.pov_relative import (  # noqa: E402
    resolve_action_mappers,
)
from concept_based_learning.pqn_visualizer import (  # noqa: E402
    _skill_input_from_obs,
    _skill_oc_psi_from_obs,
    _skill_pixel_obs_from_obs,
    align_config_with_skill_checkpoint,
    apply_skill_q_all,
    create_eval_env,
    ensure_skill_params_stack,
    label_to_skill_idx,
    load_params_from_file,
)
from concept_based_learning.environments.datatypes import (  # noqa: E402
    configure_label_routing_from_config,
)

EGO_NAMES = {
    0: "NOOP",
    1: "FIRE",
    2: "FORWARD",
    3: "RIGHT",
    4: "LEFT",
    5: "BACKWARD",
    6: "FORWARD_RIGHT",
    7: "FORWARD_LEFT",
    8: "BACKWARD_RIGHT",
    9: "BACKWARD_LEFT",
    10: "FORWARD_F",
    11: "RIGHT_F",
    12: "LEFT_F",
    13: "BACKWARD_F",
    14: "FORWARD_RIGHT_F",
    15: "FORWARD_LEFT_F",
    16: "BACKWARD_RIGHT_F",
    17: "BACKWARD_LEFT_F",
}

# Ego ids that realize as paddle UP / DOWN under Pong facing-180° profile.
# LEFT (−90°) → world DOWN → Atari RIGHT → paddle up (−y).
# RIGHT (+90°) → world UP → Atari LEFT → paddle down (+y).
STRAFE_UP_EGO = {4, 7, 9, 12, 15, 17}  # LEFT family
STRAFE_DOWN_EGO = {3, 6, 8, 11, 14, 16}  # RIGHT family
FORWARDISH = {2, 6, 7, 10, 14, 15}


def _load_config(ckpt: Path, eval_yaml: Path | None, params_tree) -> dict:
    sidecar = ckpt.with_name(ckpt.stem + "_config.yaml")
    if not sidecar.is_file():
        sidecar = ckpt.with_name(ckpt.name.replace(".safetensors", "_config.yaml"))
    with sidecar.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = flatten_experiment_cfg(raw)
    if eval_yaml is not None:
        cfg = merge_paper_eval_yaml(cfg, eval_yaml, project_root=ROOT)
    # Transfer eval defaults
    cfg["ENV_NAME"] = "pong"
    cfg["HEURISTIC_MODE"] = cfg.get("HEURISTIC_MODE") or "nearest"
    cfg.setdefault("SEED", 44)
    cfg.setdefault("MAX_OBJECTS", 2)
    cfg = align_config_with_skill_checkpoint(cfg, params_tree)
    return cfg


def _body_pose(raw_obs, objects, target_idx: int) -> tuple[float, float, float, float]:
    """Return (dist_aabb_approx_as_centre, fwd, lat, phi_deg)."""
    player = raw_obs.player
    px = float(player.x[-1] + player.width[-1] / 2.0)
    py = float(player.y[-1] + player.height[-1] / 2.0)
    tx = float(objects.x[-1, target_idx] + objects.width[-1, target_idx] / 2.0)
    ty = float(objects.y[-1, target_idx] + objects.height[-1, target_idx] / 2.0)
    dx, dy = tx - px, ty - py
    facing = resolve_facing_deg(raw_obs, env_name="pong")
    fwd, lat = world_delta_to_facing_frame(
        jnp.asarray(dx), jnp.asarray(dy), facing
    )
    phi = facing_frame_phi_rad(fwd, lat)
    dist = float(np.hypot(dx, dy))
    return dist, float(fwd), float(lat), float(np.degrees(phi))


def run_diag(
    cfg: dict,
    params_tree,
    *,
    num_episodes: int,
    max_frames: int,
    legal_mask: bool,
    mask_mode: str = "realized",
    seed: int,
) -> dict:
    cfg = dict(cfg)
    cfg["EGO_LEGAL_ACTION_MASK"] = bool(legal_mask)
    if legal_mask:
        cfg["EGO_LEGAL_ACTION_MASK_MODE"] = mask_mode
    configure_ego_obs_from_config(cfg)
    configure_label_routing_from_config(cfg)

    env, _ = create_eval_env(cfg)
    manager = EnvManager()
    components = manager.get_components("pong")
    heuristic_target = resolve_heuristic_target_fn(cfg, "pong", components)
    action_mapper, _ = resolve_action_mappers(components, cfg)
    no_target = int(components.no_target_idx)
    num_skills = int(cfg["NUM_SKILLS"])
    ego_dim = int(cfg["EGO_ACTION_DIM"])
    fire_indices = fire_ego_action_indices(ego_dim)
    legal_fn = make_legal_ego_mask_fn(
        action_mapper,
        mode=mask_mode if legal_mask else "realized",
        noop_action=int(components.ego_mapper_noop_action),
        fire_action=int(components.ego_mapper_fire_action),
        ego_action_dim=ego_dim,
    )
    skill_network = q_network_from_config(cfg)
    params_tree = ensure_skill_params_stack(params_tree, num_skills)

    ego_counts: Counter[int] = Counter()
    mapped_noop = 0
    legal_blocks_chosen = 0
    steps_total = 0
    correct_strafe = 0
    wrong_strafe = 0
    idle_when_misaligned = 0
    move_when_aligned = 0
    lat_bins = {b: Counter() for b in ABS_LAT_BUCKETS}
    signed_lat_bins = {b: Counter() for b in SIGNED_LAT_BUCKETS}
    paddle_dy_vs_lat_sign = []  # +1 if paddle moved toward ball in y
    episode_scores = []
    miss_gaps = []  # |lat| when ball crosses near paddle x (intercept attempt)

    for ep in range(num_episodes):
        rng = jax.random.PRNGKey(seed + ep * 997)
        obs, state = env.reset(rng)
        obs, state = jax.device_get((obs, state))
        prev_target = no_target
        done = False
        step = 0
        prev_py = float(obs.raw_observation.player.y[-1])
        while (not done) and step < max_frames:
            rng, step_key = jax.random.split(rng)
            target = int(
                jnp.ravel(
                    heuristic_target(
                        obs.raw_observation, obs.objects, prev_target, step_key
                    )
                )[0]
            )
            target = int(
                components.forced_target_hook(
                    jax.tree_util.tree_map(
                        lambda x: jnp.expand_dims(x, 0), obs.raw_observation
                    ),
                    jnp.array([target], dtype=jnp.int32),
                    cfg,
                )[0]
            )
            safe = 0 if target == no_target else target
            raw_label = int(obs.objects.semantic_label[-1, safe])
            sk_idx = int(label_to_skill_idx(raw_label, num_skills))
            ego_obs = _skill_input_from_obs(
                cfg, obs.raw_observation, obs.objects, safe, None
            )
            pixel_obs = _skill_pixel_obs_from_obs(cfg, obs)
            q_all = apply_skill_q_all(
                skill_network,
                params_tree,
                ego_obs,
                target_idx=safe,
                pixel_obs=pixel_obs,
                objects=obs.objects,
                raw_obs=obs.raw_observation,
                oc_psi=_skill_oc_psi_from_obs(
                    cfg,
                    obs.raw_observation,
                    obs.objects,
                    safe,
                    int(cfg.get("MAX_OBJECTS", 2)),
                ),
                stack_size=int(cfg.get("FRAME_STACK_SIZE", 4)),
                destroy_skill_idx=0,
            )
            q = q_all[sk_idx]
            # COLLECT should not fire; keep fire available but note skill
            legal = legal_fn(
                obs.raw_observation, jnp.int32(safe), obs.objects
            )
            unmasked_ego = int(jnp.argmax(q))
            if legal_mask:
                q = mask_q_with_legal_ego(q, legal)
            chosen = int(jnp.argmax(q))
            if not bool(legal[unmasked_ego]) and chosen != unmasked_ego:
                legal_blocks_chosen += 1

            game_a = int(
                action_mapper(
                    jnp.int32(chosen),
                    obs.raw_observation,
                    jnp.int32(target),
                    obs.objects,
                )
            )
            if target == no_target:
                game_a = 0

            dist, fwd, lat, phi_deg = _body_pose(obs.raw_observation, obs.objects, safe)
            ego_counts[chosen] += 1
            steps_total += 1
            if game_a == 0:
                mapped_noop += 1

            abs_lat = abs(lat)
            if abs_lat < 8:
                bucket = "|lat|<8"
            elif abs_lat <= 20:
                bucket = "|lat|8-20"
            else:
                bucket = "|lat|>20"
            lat_bins[bucket][chosen] += 1
            signed_lat_bins[signed_lat_bucket(lat)][chosen] += 1

            # Correct strafe: lat>0 (ball below) → RIGHT family; lat<0 → LEFT family
            if abs_lat >= 8:
                want_down = lat > 0
                is_down = chosen in STRAFE_DOWN_EGO
                is_up = chosen in STRAFE_UP_EGO
                if want_down and is_down:
                    correct_strafe += 1
                elif (not want_down) and is_up:
                    correct_strafe += 1
                elif is_up or is_down:
                    wrong_strafe += 1
                else:
                    idle_when_misaligned += 1
            else:
                if chosen in STRAFE_UP_EGO or chosen in STRAFE_DOWN_EGO:
                    move_when_aligned += 1

            # Near-paddle miss gap: ball within ~20px of paddle in x
            px = float(
                obs.raw_observation.player.x[-1]
                + obs.raw_observation.player.width[-1] / 2.0
            )
            tx = float(obs.objects.x[-1, safe] + obs.objects.width[-1, safe] / 2.0)
            if abs(tx - px) < 20.0 and abs_lat >= 4:
                miss_gaps.append(abs_lat)

            obs, state, reward, done, _info = env.step(state, game_a)
            obs, state = jax.device_get((obs, state))
            done = bool(done)
            py = float(obs.raw_observation.player.y[-1])
            dpy = py - prev_py
            if abs_lat >= 8 and abs(dpy) > 0.1:
                # Toward ball: if lat>0 need +dy; if lat<0 need −dy
                toward = (lat > 0 and dpy > 0) or (lat < 0 and dpy < 0)
                paddle_dy_vs_lat_sign.append(1 if toward else 0)
            prev_py = py
            prev_target = target
            step += 1

        # Prefer env score if available
        try:
            es = state.env_state
            ps = float(getattr(es, "player_score", 0))
            ene = float(getattr(es, "enemy_score", 0))
            episode_scores.append({"player": ps, "enemy": ene, "diff": ps - ene})
        except Exception:
            episode_scores.append({"player": float("nan"), "enemy": float("nan"), "diff": float("nan")})

    named = {EGO_NAMES.get(k, str(k)): int(v) for k, v in ego_counts.most_common()}
    forwardish = sum(ego_counts[i] for i in FORWARDISH)
    strafe = sum(ego_counts[i] for i in (STRAFE_UP_EGO | STRAFE_DOWN_EGO))
    toward_frac = (
        float(np.mean(paddle_dy_vs_lat_sign)) if paddle_dy_vs_lat_sign else float("nan")
    )
    abs_tv = mean_pairwise_tv(lat_bins, n_actions=ego_dim, min_count=16)
    signed_tv = mean_pairwise_tv(signed_lat_bins, n_actions=ego_dim, min_count=16)
    return {
        "legal_mask": legal_mask,
        "mask_mode": mask_mode if legal_mask else None,
        "steps": steps_total,
        "episodes": num_episodes,
        "episode_scores": episode_scores,
        "mean_score_diff": float(np.nanmean([e["diff"] for e in episode_scores])),
        "ego_histogram": named,
        "forwardish_frac": forwardish / max(1, steps_total),
        "strafe_frac": strafe / max(1, steps_total),
        "mapped_noop_frac": mapped_noop / max(1, steps_total),
        "legal_blocks_unmasked_argmax": legal_blocks_chosen,
        "legal_blocks_frac": legal_blocks_chosen / max(1, steps_total),
        "correct_strafe": correct_strafe,
        "wrong_strafe": wrong_strafe,
        "idle_when_misaligned": idle_when_misaligned,
        "move_when_aligned": move_when_aligned,
        "strafe_correct_rate": correct_strafe
        / max(1, correct_strafe + wrong_strafe),
        "idle_rate_when_misaligned": idle_when_misaligned
        / max(1, correct_strafe + wrong_strafe + idle_when_misaligned),
        "paddle_moves_toward_ball_frac": toward_frac,
        "n_paddle_moves_logged": len(paddle_dy_vs_lat_sign),
        "mean_abs_lat_near_paddle": float(np.mean(miss_gaps)) if miss_gaps else float("nan"),
        "n_near_paddle_samples": len(miss_gaps),
        "bearing_tv_abs_mean": abs_tv["bearing_tv_mean"],
        "bearing_tv_abs_max": abs_tv["bearing_tv_max"],
        "bearing_tv_signed_mean": signed_tv["bearing_tv_mean"],
        "bearing_tv_signed_max": signed_tv["bearing_tv_max"],
        "ego_by_lat_bucket": {
            b: {EGO_NAMES.get(k, str(k)): int(v) for k, v in c.most_common(8)}
            for b, c in lat_bins.items()
        },
        "ego_by_signed_lat_bucket": {
            b: {EGO_NAMES.get(k, str(k)): int(v) for k, v in c.most_common(8)}
            for b, c in signed_lat_bins.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--eval-yaml", type=Path, default=None)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument(
        "--modes",
        default="both",
        choices=("off", "on", "both"),
        help="Legal-mask modes to run",
    )
    ap.add_argument(
        "--mask-mode",
        default="realized",
        choices=("realized", "pure_cardinal", "strafe_only"),
        help="When legal mask is on: realized vs pure_cardinal (no diagonals / move+FIRE)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"Loading {args.ckpt}", flush=True)
    params = load_params_from_file(str(args.ckpt))
    cfg = _load_config(args.ckpt, args.eval_yaml, params)
    print(
        f"  POV mapper={cfg.get('USE_POV_RELATIVE_MAPPER')} "
        f"facing_frame={cfg.get('USE_POV_FACING_FRAME')} "
        f"psi_bins={cfg.get('EGO_PSI_ANGLE_BINS')} "
        f"clip={cfg.get('EGO_DISTANCE_CLIP_MAX_PX')}",
        flush=True,
    )
    modes = []
    if args.modes in ("off", "both"):
        modes.append(False)
    if args.modes in ("on", "both"):
        modes.append(True)

    results = {"ckpt": str(args.ckpt), "eval_yaml": str(args.eval_yaml), "runs": []}
    for legal in modes:
        print(f"\n=== legal_mask={legal} ===", flush=True)
        stats = run_diag(
            cfg,
            params,
            num_episodes=args.episodes,
            max_frames=args.max_frames,
            legal_mask=legal,
            mask_mode=args.mask_mode,
            seed=args.seed,
        )
        results["runs"].append(stats)
        print(json.dumps(stats, indent=2), flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
