#!/usr/bin/env python3
"""Instrument a frozen Seaquest skill checkpoint (no retraining).

Separates three failure modes on ``no_enemies`` (and optionally ``base``):

1. Real pickups / waypoint arrivals without COLLECT feedback → event detection.
2. Six divers carried but surface waypoint not selected → selector / overrides.
3. Waypoint selected but skill stalls → COLLECT execution / training data.

Default episodes match the avsep seed-44 mod-eval seed formula
``ep_seed = eval_seed + seed_idx * 10000 + episode_idx * 997`` with
``eval_seed=10044``:

  low   seed_idx=2 ep=0 → 30044 (return 1460 in the finished eval)
  mid   seed_idx=0 ep=0 → 10044 (return 3700)
  high  seed_idx=1 ep=0 → 20044 (return 58400)

Example (cluster1, GPU 0)::

    source /tmp/cbl_queue_env.sh
    export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled
    $PY -u scripts/diagnose_seaquest_collect_events.py \\
      --model outputs/models/paper_eval/thesis_final/tf_seaquest_skills_ego_avsep_seed44/seaquest/hierarchical_pqn_seaquest_seed44.safetensors \\
      --paper-config configs/experiment/paper_eval/thesis_final/tf_seaquest_skills_ego_avsep_seed44_eval.yaml \\
      --game-mods no_enemies \\
      --out-dir tmp/avsep_collect_diag \\
      --max-frames 12000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Headless-safe before pygame import chains.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("WANDB_MODE", "disabled")

import jax
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from concept_based_learning.baselines.fixed_ego_skills import (  # noqa: E402
    select_fixed_ego_action,
    uses_fixed_skill_policy,
)
from concept_based_learning.environments.datatypes import (  # noqa: E402
    LABEL_COLLECT,
    label_to_skill_idx,
)
from concept_based_learning.environments.env_manager import (  # noqa: E402
    EnvManager,
    resolve_no_target_game_action,
)
from concept_based_learning.environments.heuristics import (  # noqa: E402
    initial_policy_state,
    resolve_heuristic_target_fn,
    resolve_heuristic_with_mission_fn,
    resolve_heuristic_with_state_fn,
)
from concept_based_learning.environments.meta_target_mask import (  # noqa: E402
    meta_fire_target_idx as resolve_meta_fire_target_idx,
    meta_noop_target_idx as resolve_meta_noop_target_idx,
)
from concept_based_learning.environments.skill_rewards import (  # noqa: E402
    explain_collect_contact,
    extract_skill_rewards,
)
from concept_based_learning.networks.skill_network import q_network_from_config  # noqa: E402
from concept_based_learning.observations.action_space import (  # noqa: E402
    fire_ego_action_indices,
    legal_ego_mask_mode_from_config,
    make_legal_ego_mask_fn,
    mask_q_with_legal_ego,
)
from concept_based_learning.observations.egocentric import (  # noqa: E402
    configure_ego_obs_from_config,
)
from concept_based_learning.observations.pov_relative import (  # noqa: E402
    resolve_action_mappers,
)
from concept_based_learning.pqn_visualizer import (  # noqa: E402
    _destroy_skill_idx,
    _is_idle_meta_target,
    _skill_input_from_obs,
    _skill_oc_psi_from_obs,
    _skill_pixel_obs_from_obs,
    align_config_with_skill_checkpoint,
    apply_skill_q_all,
    create_eval_env,
    ensure_skill_params_stack,
    load_base_config,
    load_config_for_checkpoint,
    load_params_from_file,
    resolve_tandem_checkpoint_paths,
    validate_skill_checkpoint_ego_dim,
)
from concept_based_learning.training.episode_stats import (  # noqa: E402
    unwrap_base_env_state,
)

# Seaquest append-NOOP layout (heuristic_policy docstring).
SURFACE_WAYPOINT_IDX = 33
SKILL_NAMES = ("DESTROY", "COLLECT", "AVOID", "NAVIGATE")


def _leaf(x):
    a = np.asarray(jax.device_get(x))
    if a.ndim == 0:
        return a.item()
    return a.reshape(-1)[-1].item()


def _get_target_obj(objects, idx: int):
    return jax.tree_util.tree_map(lambda x: jnp.take(x, int(idx), axis=-1), objects)


def _episode_seed(eval_seed: int, seed_idx: int, episode_idx: int) -> int:
    return int(eval_seed) + int(seed_idx) * 10_000 + int(episode_idx) * 997


def _load_config(model: Path, paper_config: Path | None, game_mods: list[str] | None):
    base = load_base_config()
    skill_path, _meta = resolve_tandem_checkpoint_paths(str(model), None)
    if skill_path is None:
        raise SystemExit(f"Not a skill checkpoint: {model}")
    base, _ = load_config_for_checkpoint(skill_path, base)
    params = load_params_from_file(skill_path)
    if paper_config is not None:
        from concept_based_learning.paper_eval.config_merge import merge_paper_eval_yaml

        base = merge_paper_eval_yaml(base, str(paper_config), project_root=_ROOT)
    if game_mods is not None:
        base["GAME_MODS"] = list(game_mods)
        base["EVAL_GAME_MODS"] = list(game_mods)
    from concept_based_learning.paper_eval.eval_harness import (
        _overlay_ego_flags_from_skill_config,
    )

    base = _overlay_ego_flags_from_skill_config(base, skill_path)
    base = align_config_with_skill_checkpoint(base, params)
    base["WANDB_MODE"] = "disabled"
    return base, params


def _run_episode(*, config, params_tree, ep_seed: int, max_frames: int, out_jsonl: Path):
    from concept_based_learning.environments.datatypes import (
        configure_label_routing_from_config,
    )

    configure_ego_obs_from_config(config)
    configure_label_routing_from_config(config)
    env, _ = create_eval_env(config)
    env_name = config.get("ENV_NAME", "seaquest")
    components = EnvManager().get_components(env_name)
    heuristic_target = resolve_heuristic_target_fn(config, env_name, components)
    heuristic_with_mission = resolve_heuristic_with_mission_fn(config, env_name)
    heuristic_with_state = resolve_heuristic_with_state_fn(config, env_name, components)
    no_target_idx = int(components.no_target_idx)
    action_mapper, _ = resolve_action_mappers(components, config)
    forced_target_hook = components.forced_target_hook
    extract_life_lost = components.extract_life_lost
    reset_policy_on_life_lost = bool(
        getattr(components, "reset_policy_on_life_lost", False)
    )
    num_skills = int(config["NUM_SKILLS"])
    ego_action_dim = int(config["EGO_ACTION_DIM"])
    skill_network = q_network_from_config(config)
    fixed_skill_policy = uses_fixed_skill_policy(config)
    use_legal = bool(config.get("EGO_LEGAL_ACTION_MASK", False))
    legal_fn = make_legal_ego_mask_fn(
        action_mapper,
        mode=legal_ego_mask_mode_from_config(config) or "realized",
        noop_action=int(components.ego_mapper_noop_action),
        fire_action=int(components.ego_mapper_fire_action),
        ego_action_dim=ego_action_dim,
    )
    meta_noop_idx = int(resolve_meta_noop_target_idx(config))
    meta_use_noop = bool(config.get("META_USE_NOOP_TARGET", False))
    meta_fire_idx = int(resolve_meta_fire_target_idx(config))
    meta_use_fire = bool(config.get("META_USE_FIRE_TARGET", False))
    meta_fire_action = int(config.get("META_FIRE_GAME_ACTION", 1))
    restrict_fire = bool(config.get("RESTRICT_FIRE_FOR_NON_DESTROY", False))

    params_tree = ensure_skill_params_stack(params_tree, num_skills)
    rng = jax.random.PRNGKey(ep_seed)
    obs, state = env.reset(rng)
    obs, state = jax.device_get((obs, state))
    dummy = _skill_input_from_obs(
        config, obs.raw_observation, obs.objects, jnp.int32(0)
    )
    validate_skill_checkpoint_ego_dim(config, int(dummy.shape[-1]))

    def _apply_all_skills(
        params_stack, ego_batch, target_idx_arr, pixel_batch, objects_batch, raw_batch, oc_psi
    ):
        return apply_skill_q_all(
            skill_network,
            params_stack,
            ego_batch,
            target_idx=target_idx_arr,
            pixel_obs=pixel_batch,
            objects=objects_batch,
            raw_obs=raw_batch,
            oc_psi=oc_psi,
            stack_size=int(config.get("FRAME_STACK_SIZE", 4)),
            destroy_skill_idx=_destroy_skill_idx(config),
        )

    apply_q = jax.jit(_apply_all_skills)

    prev_target_idx = no_target_idx
    mission_target_idx = no_target_idx
    policy_state = initial_policy_state(no_target_idx)
    prev_obs = obs

    ep_return = 0.0
    pickups = 0
    rescues = 0
    life_losses = []
    collect_events = []
    missed_deactivations = []
    full_rack_cycles = []
    pending_full_at = None
    pending_wp_at = None
    steps = 0

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as fh:
        for step_i in range(max_frames):
            rng, step_key = jax.random.split(rng)
            raw = obs.raw_observation
            objects = obs.objects

            if heuristic_with_state is not None:
                target_obj_idx, policy_state = heuristic_with_state(
                    raw,
                    objects,
                    jnp.array(prev_target_idx, dtype=jnp.int32),
                    policy_state,
                    step_key,
                )
                target_obj_idx = int(jnp.ravel(target_obj_idx)[0])
            elif heuristic_with_mission is not None:
                target_obj_idx, mission_target_idx = heuristic_with_mission(
                    raw,
                    objects,
                    jnp.array(prev_target_idx, dtype=jnp.int32),
                    jnp.array(mission_target_idx, dtype=jnp.int32),
                    step_key,
                )
                target_obj_idx = int(jnp.ravel(target_obj_idx)[0])
                mission_target_idx = int(jnp.ravel(mission_target_idx)[0])
            else:
                target_obj_idx = heuristic_target(
                    raw, objects, prev_target_idx, step_key
                )
                target_obj_idx = int(jnp.ravel(target_obj_idx)[0])
            target_obj_idx = int(
                forced_target_hook(
                    jax.tree_util.tree_map(
                        lambda x: jnp.expand_dims(x, 0), raw
                    ),
                    jnp.asarray([target_obj_idx], dtype=jnp.int32),
                    config,
                )[0]
            )
            prev_target_idx = target_obj_idx

            is_idle = _is_idle_meta_target(
                target_obj_idx,
                no_target_idx=no_target_idx,
                meta_noop_idx=meta_noop_idx,
                meta_use_noop=meta_use_noop,
                max_objects=int(config.get("MAX_OBJECTS", 35)),
            )
            is_meta_fire = meta_use_fire and target_obj_idx == meta_fire_idx
            chosen_ego = 0
            if is_idle:
                global_action = (
                    0
                    if int(target_obj_idx) == int(meta_noop_idx)
                    or int(target_obj_idx) == int(config.get("MAX_OBJECTS", 35)) - 1
                    else resolve_no_target_game_action(config, components)
                )
                active_skill = -1
                label = -1
            elif is_meta_fire:
                global_action = meta_fire_action
                active_skill = 0
                label = 0
            else:
                label = int(jax.device_get(objects.semantic_label[-1, target_obj_idx]))
                active_skill = int(label_to_skill_idx(label, num_skills))
                ego = _skill_input_from_obs(
                    config, raw, objects, jnp.int32(target_obj_idx)
                )
                ego_b = ego[jnp.newaxis, ...]
                t_arr = jnp.asarray([target_obj_idx], dtype=jnp.int32)
                pixel = _skill_pixel_obs_from_obs(config, obs)
                pixel_b = jnp.expand_dims(pixel, 0) if pixel is not None else None
                oc_psi = _skill_oc_psi_from_obs(
                    config,
                    raw,
                    objects,
                    t_arr[0],
                    int(config.get("MAX_OBJECTS", 35)),
                )
                if fixed_skill_policy:
                    chosen_ego = int(
                        select_fixed_ego_action(
                            config, active_skill, raw, objects, t_arr[0]
                        )
                    )
                else:
                    q_vals = apply_q(
                        params_tree, ego_b, t_arr, pixel_b, objects, raw, oc_psi
                    )
                    q_vals = jnp.squeeze(q_vals, axis=1)
                    q_active = q_vals[active_skill]
                    if restrict_fire and active_skill != 0:
                        q_active = q_active.at[
                            fire_ego_action_indices(ego_action_dim)
                        ].set(-jnp.inf)
                    if use_legal:
                        legal = legal_fn(raw, t_arr[0], objects)
                        q_active = mask_q_with_legal_ego(q_active, legal)
                    chosen_ego = int(jnp.argmax(q_active))
                global_action = int(
                    action_mapper(chosen_ego, raw, t_arr[0], objects)
                )

            player_x = float(_leaf(raw.player.x))
            player_y = float(_leaf(raw.player.y))
            oxygen = float(_leaf(raw.oxygen_level))
            divers = int(_leaf(raw.collected_divers))
            just_surfaced = int(_leaf(getattr(raw, "just_surfaced", 0)))
            selecting_surface_wp = int(target_obj_idx) == SURFACE_WAYPOINT_IDX

            next_obs, next_state, reward, done, _info = env.step(state, global_action)
            next_obs, next_state = jax.device_get((next_obs, next_state))
            done = bool(done)
            reward = float(np.asarray(reward).reshape(-1)[0])
            ep_return += reward
            life_lost = 0
            if reset_policy_on_life_lost:
                life_lost = int(
                    jax.device_get(extract_life_lost(state, next_state))
                )

            next_divers = int(_leaf(next_obs.raw_observation.collected_divers))
            next_oxygen = float(_leaf(next_obs.raw_observation.oxygen_level))
            next_player_y = float(_leaf(next_obs.raw_observation.player.y))
            prev_gs = unwrap_base_env_state(state)
            next_gs = unwrap_base_env_state(next_state)
            prev_rescues = int(_leaf(getattr(prev_gs, "successful_rescues", 0)))
            next_rescues = int(_leaf(getattr(next_gs, "successful_rescues", 0)))
            next_just_surfaced = int(_leaf(getattr(next_gs, "just_surfaced", 0)))
            pickup = next_divers > divers
            rescue = next_rescues > prev_rescues
            if pickup:
                pickups += 1
            if rescue:
                rescues += 1

            safe_t = target_obj_idx if not is_idle else 0
            prev_t = _get_target_obj(prev_obs.objects, safe_t)
            next_t = _get_target_obj(next_obs.objects, safe_t)
            explain = explain_collect_contact(
                prev_obs.raw_observation,
                next_obs.raw_observation,
                prev_t,
                next_t,
            )
            skill_r, skill_d = extract_skill_rewards(
                prev_obs.raw_observation,
                next_obs.raw_observation,
                prev_t,
                next_t,
                jnp.int32(life_lost),
                config=config,
            )
            skill_r = np.asarray(jax.device_get(skill_r))
            skill_d = np.asarray(jax.device_get(skill_d))
            collect_r = float(skill_r[1])
            collect_done = bool(skill_d[1])

            if explain["reason"] == "missed_deactivation":
                missed_deactivations.append(step_i)
            if collect_done or explain["success"] or pickup or explain["reason"] != "none":
                collect_events.append(
                    {
                        "step": step_i,
                        "pickup_env": pickup,
                        "collect_reward": collect_r,
                        "collect_done": collect_done,
                        "explain": explain,
                        "target_idx": int(target_obj_idx),
                        "skill": SKILL_NAMES[active_skill]
                        if 0 <= active_skill < len(SKILL_NAMES)
                        else str(active_skill),
                        "label": int(label),
                        "divers_before": divers,
                        "divers_after": next_divers,
                    }
                )

            if pending_full_at is None and next_divers >= 6 and divers < 6:
                pending_full_at = step_i
                pending_wp_at = None
            if (
                pending_full_at is not None
                and pending_wp_at is None
                and selecting_surface_wp
            ):
                pending_wp_at = step_i
            if pending_full_at is not None and rescue:
                full_rack_cycles.append(
                    {
                        "full_at": pending_full_at,
                        "surface_wp_at": pending_wp_at,
                        "rescue_at": step_i,
                        "steps_full_to_wp": (
                            None
                            if pending_wp_at is None
                            else pending_wp_at - pending_full_at
                        ),
                        "steps_wp_to_rescue": (
                            None
                            if pending_wp_at is None
                            else step_i - pending_wp_at
                        ),
                        "steps_full_to_rescue": step_i - pending_full_at,
                    }
                )
                pending_full_at = None
                pending_wp_at = None

            if life_lost > 0 or (done and next_oxygen <= 0.0):
                if next_oxygen <= 0.0 or oxygen <= 0.0:
                    cause = "oxygen_depletion"
                elif next_player_y <= 46 and next_divers < 6 and next_oxygen > 0:
                    cause = "unsuccessful_surface"
                else:
                    cause = "other_or_enemy"
                life_losses.append(
                    {
                        "step": step_i,
                        "life_lost": life_lost,
                        "done": done,
                        "cause": cause,
                        "oxygen": next_oxygen,
                        "divers": next_divers,
                        "player_y": next_player_y,
                    }
                )

            row = {
                "step": step_i,
                "target_idx": int(target_obj_idx),
                "selecting_surface_wp": selecting_surface_wp,
                "label": int(label),
                "skill": SKILL_NAMES[active_skill]
                if 0 <= active_skill < len(SKILL_NAMES)
                else str(active_skill),
                "skill_idx": int(active_skill),
                "ego_action": int(chosen_ego),
                "game_action": int(global_action),
                "player_x": player_x,
                "player_y": player_y,
                "oxygen": oxygen,
                "divers": divers,
                "just_surfaced": just_surfaced,
                "env_reward": reward,
                "done": done,
                "life_lost": int(life_lost),
                "pickup_env": pickup,
                "rescue_env": rescue,
                "collect_reward": collect_r,
                "collect_done": collect_done,
                "collect_reason": explain["reason"],
                "collect_success": explain["success"],
                "was_collected_stamp": explain["was_collected"],
                "missed_deactivation": explain["reason"] == "missed_deactivation",
                "destroy_reward": float(skill_r[0]),
                "avoid_reward": float(skill_r[2]),
            }
            fh.write(json.dumps(row) + "\n")

            steps = step_i + 1
            prev_obs = next_obs
            obs, state = next_obs, next_state
            if done:
                break
            if step_i % 500 == 0:
                print(
                    f"  seed={ep_seed} step={step_i} return={ep_return:.0f} "
                    f"divers={divers} pickups={pickups} rescues={rescues}",
                    flush=True,
                )

    pickup_without_collect = [
        e
        for e in collect_events
        if e["pickup_env"] and not (e["collect_done"] or e["collect_reward"] >= 1.0)
    ]
    collect_without_pickup = [
        e
        for e in collect_events
        if (e["collect_done"] or e["collect_reward"] >= 1.0)
        and not e["pickup_env"]
        and int(e["explain"]["label"]) == int(LABEL_COLLECT)
        and e["target_idx"] != SURFACE_WAYPOINT_IDX
    ]
    open_full_rack = None
    if pending_full_at is not None:
        open_full_rack = {
            "full_at": pending_full_at,
            "surface_wp_at": pending_wp_at,
            "note": "ended episode still carrying a full rack",
        }

    return {
        "ep_seed": ep_seed,
        "steps": steps,
        "return": ep_return,
        "pickups": pickups,
        "rescues": rescues,
        "life_losses": life_losses,
        "missed_deactivation_steps": missed_deactivations,
        "n_missed_deactivation": len(missed_deactivations),
        "full_rack_cycles": full_rack_cycles,
        "open_full_rack": open_full_rack,
        "pickup_without_collect_feedback": pickup_without_collect,
        "collect_success_without_env_pickup": collect_without_pickup,
        "n_collect_events_logged": len(collect_events),
        "jsonl": str(out_jsonl),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--paper-config", type=str, default=None)
    p.add_argument("--game-mods", nargs="*", default=["no_enemies"])
    p.add_argument("--eval-seed", type=int, default=10044)
    p.add_argument(
        "--episodes",
        type=str,
        default="2:0,0:0,1:0",
        help="Comma list of seed_idx:episode_idx (mod-eval indexing).",
    )
    p.add_argument("--ep-seeds", type=str, default=None)
    p.add_argument("--max-frames", type=int, default=12000)
    p.add_argument("--out-dir", type=str, default="tmp/avsep_collect_diag")
    args = p.parse_args()

    model = Path(args.model)
    if not model.is_file():
        alt = _ROOT / args.model
        model = alt if alt.is_file() else model
    if not model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    paper = Path(args.paper_config) if args.paper_config else None
    if paper is not None and not paper.is_file():
        alt = _ROOT / args.paper_config
        paper = alt if alt.is_file() else paper

    config, params = _load_config(
        model, paper, list(args.game_mods) if args.game_mods is not None else None
    )
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ep_seeds:
        episode_meta = [{"ep_seed": int(s)} for s in args.ep_seeds.split(",") if s.strip()]
    else:
        episode_meta = []
        for part in args.episodes.split(","):
            a, b = part.split(":")
            si, ei = int(a), int(b)
            episode_meta.append(
                {
                    "seed_idx": si,
                    "episode_idx": ei,
                    "ep_seed": _episode_seed(args.eval_seed, si, ei),
                }
            )

    summaries = []
    for meta in episode_meta:
        ep_seed = int(meta["ep_seed"])
        tag = f"seed{ep_seed}"
        if "seed_idx" in meta:
            tag = f"s{meta['seed_idx']}_e{meta['episode_idx']}_seed{ep_seed}"
        print(f"\n=== episode {tag} (max_frames={args.max_frames}) ===", flush=True)
        summary = _run_episode(
            config=config,
            params_tree=params,
            ep_seed=ep_seed,
            max_frames=int(args.max_frames),
            out_jsonl=out_dir / f"{tag}.jsonl",
        )
        summary["meta"] = meta
        summaries.append(summary)
        (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f"  done return={summary['return']:.0f} pickups={summary['pickups']} "
            f"rescues={summary['rescues']} missed_deact={summary['n_missed_deactivation']} "
            f"cycles={len(summary['full_rack_cycles'])}",
            flush=True,
        )

    report = {
        "model": str(model),
        "paper_config": str(paper) if paper else None,
        "game_mods": list(args.game_mods) if args.game_mods is not None else [],
        "max_frames": int(args.max_frames),
        "episodes": summaries,
        "decision_table": {
            "pickup_without_collect_feedback": "Repair event detection before reward changes.",
            "full_rack_but_surface_wp_never_selected": "Inspect selector / overrides.",
            "surface_wp_selected_but_slow_or_no_rescue": "Investigate COLLECT execution / training data.",
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# Seaquest collect-event diagnosis",
        "",
        f"Model: `{model}`",
        f"Mods: `{report['game_mods']}`",
        "",
        "| episode | return | pickups | rescues | missed_deact | full→wp | wp→rescue | pickup w/o COLLECT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        cycles = s["full_rack_cycles"]
        f2w = (
            ",".join(
                str(c["steps_full_to_wp"]) if c["steps_full_to_wp"] is not None else "never"
                for c in cycles
            )
            or "—"
        )
        w2r = (
            ",".join(
                str(c["steps_wp_to_rescue"]) if c["steps_wp_to_rescue"] is not None else "?"
                for c in cycles
            )
            or "—"
        )
        md.append(
            f"| {s['ep_seed']} | {s['return']:.0f} | {s['pickups']} | {s['rescues']} | "
            f"{s['n_missed_deactivation']} | {f2w} | {w2r} | "
            f"{len(s['pickup_without_collect_feedback'])} |"
        )
    md.extend(
        [
            "",
            "## Decision table",
            "",
            "| Observation | Next action |",
            "|---|---|",
            "| Real pickups without COLLECT feedback | Repair event detection before changing rewards. |",
            "| Six divers carried but waypoint not selected | Inspect the selector and its overrides. |",
            "| Waypoint selected but skill stalls | Investigate COLLECT execution and its training data. |",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {out_dir / 'report.json'} and {out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
