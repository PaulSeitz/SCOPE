#!/usr/bin/env python3
"""Eval-only: does an ``ego_abs_pos`` checkpoint use its absolute-position input?

For one trained stem, runs base + layout mods under: control (real position),
frozen (reset position in every frame), mirror_x (x -> 1 - x). Uses the normal eval
harness with the eval YAML's seeds. Writes only to
``outputs/models/paper_eval/eval/abspos_intervention/``.

    python scripts/run_abspos_intervention.py tf_seaquest_skills_ego_abspos_seed44
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]

import run_paper_eval as rpe  # noqa: E402
assert Path(rpe.__file__).resolve().parent == ROOT / "scripts", rpe.__file__
from concept_based_learning.environments.env_manager import EnvManager  # noqa: E402
from concept_based_learning.observations.egocentric import (  # noqa: E402
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    _center_y,
    _get_player,
    _uses_bottom_edge_y,
    configure_ego_obs_from_config,
)
from concept_based_learning.paper_eval.eval_harness import run_mod_eval  # noqa: E402

MOD_KEYS = {
    "seaquest": ["base", "shift_lanes", "random_spawns"],
    "namethisgame": ["base", "shift_tentacle_spawn_x"],
}


def reset_position(cfg: dict, env_name: str) -> tuple[float, float]:
    """Normalized player centre after a reset of the unmodified game."""
    configure_ego_obs_from_config(cfg)
    env, _ = EnvManager().make_env(env_name, dict(cfg), train=False)
    obs, _ = env.reset(jax.random.PRNGKey(0))
    p = _get_player(obs.raw_observation)
    x = (float(p.x[-1]) + float(p.width[-1]) / 2) / SCREEN_WIDTH
    y = float(
        _center_y(p.y[-1], p.height[-1], y_is_bottom_edge=_uses_bottom_edge_y(obs.raw_observation))
    ) / SCREEN_HEIGHT
    return x, y


def main(stem: str) -> None:
    path = ROOT / "configs/experiment/paper_eval/context_ablation" / f"{stem}_eval.yaml"
    raw = yaml.safe_load(path.read_text())
    if raw["SKILL_INPUT_MODE"] != "ego_abs_pos":
        raise SystemExit(f"{stem} is not an ego_abs_pos arm")
    env_name = raw["ENV_NAME"]
    cfg = rpe._architecture_config(raw)
    cfg["ENV_NAME"] = env_name
    rpe._pin_jaxatari_action_sets_to_host()
    x, y = reset_position(cfg, env_name)
    out_dir = ROOT / "outputs/models/paper_eval/eval/abspos_intervention"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in (("control", ""), ("frozen", f"frozen:{x:.6f},{y:.6f}"), ("mirror_x", "mirror_x")):
        out = out_dir / f"{stem}_{name}_mods.json"
        if out.is_file():
            print(f"skip existing {out.name}", flush=True)
            continue
        os.environ["CBL_ABS_POS_INTERVENTION"] = spec
        print(f"=== {stem} {name} spec={spec!r}", flush=True)
        run_mod_eval(
            agent_kind="skills",
            config=cfg,
            env_name=env_name,
            skills_path=raw["PAPER_EVAL_SKILLS_CKPT"],
            mod_keys=MOD_KEYS[env_name],
            mods_allowlist=str(raw.get("PAPER_EVAL_MODS") or env_name),
            num_episodes=3,
            num_seeds=int(raw["EVAL_NUM_SEEDS"]),
            base_seed=int(raw["EVAL_SEED"]),
            max_frames=rpe.SUITE_EVAL_MAX_FRAMES,
            output_json=out,
            extra_meta={"abs_pos_intervention": name, "abs_pos_spec": spec, "config_stem": path.stem},
        )
    print(f"INTERVENTION_DONE {stem}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
