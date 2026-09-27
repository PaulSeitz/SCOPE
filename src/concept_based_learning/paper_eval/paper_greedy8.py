"""PQN-paper style greedy eval: 8 fixed seeds × 1 episode × 27k agent frames.

Mirrors Gallici et al. Atari reporting (ε=0, long episodes ≈108k ALE frames at
skip 4). Used as an extra metric alongside home ``paper_core`` mod CSV evals.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np

from concept_based_learning.baselines.flat_pqn import (
    QNetwork,
    _make_jitted_flat_pqn_episode_runner,
    flat_pqn_network_kwargs,
    infer_flat_pqn_action_dim,
    infer_flat_pqn_object_centric,
    load_flat_pqn_params,
    load_flat_pqn_training_config,
    make_flat_pqn_eval_env,
)
from concept_based_learning.pqn_visualizer import resolve_config_path

# Clean primes; deliberately not the training seeds 44/45/46.
DEFAULT_PAPER_GREEDY_SEEDS: tuple[int, ...] = (7, 11, 13, 17, 19, 23, 29, 31)
DEFAULT_PAPER_GREEDY_MAX_FRAMES = 27_000  # ≈108k ALE frames @ skip 4
PAPER_SEAQUEST_PQN_REF = 15898.9  # Table 9 @ 400M frames


def run_paper_greedy8_eval(
    model_path: str | Path,
    *,
    eval_seeds: Sequence[int] = DEFAULT_PAPER_GREEDY_SEEDS,
    max_frames: int = DEFAULT_PAPER_GREEDY_MAX_FRAMES,
    env_name: str = "seaquest",
    progress: bool = True,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Greedy flat-PQN eval: one episode per seed, fixed PRNG keys."""
    resolved = resolve_config_path(str(model_path))
    if not resolved or not Path(resolved).is_file():
        raise FileNotFoundError(f"Flat PQN checkpoint not found: {model_path}")

    config = load_flat_pqn_training_config(resolved)
    params = load_flat_pqn_params(resolved)
    bs_path = resolved.replace(".safetensors", "_bs.safetensors")
    batch_stats = load_flat_pqn_params(bs_path) if Path(bs_path).is_file() else {}
    object_centric = infer_flat_pqn_object_centric(resolved, config)

    env = make_flat_pqn_eval_env(
        object_centric=object_centric,
        game_mods=[],
        config=config,
        env_name=env_name,
        required_action_dim=infer_flat_pqn_action_dim(params),
    )
    network = QNetwork(
        **flat_pqn_network_kwargs(
            config,
            params,
            action_dim=int(env.action_space().n),
            object_centric=object_centric,
        )
    )
    run_episode = _make_jitted_flat_pqn_episode_runner(
        env,
        network,
        params,
        batch_stats,
        max_frames=int(max_frames),
        env_name=str(env_name).lower(),
    )

    seeds = [int(s) for s in eval_seeds]
    if progress:
        print(
            f"  Paper greedy-8: compile (max_frames={max_frames}, "
            f"n_seeds={len(seeds)}, native={config.get('USE_NATIVE_DOWNSCALING')})...",
            flush=True,
        )
    t0 = time.time()
    _ = jax.block_until_ready(run_episode(jax.random.PRNGKey(0))[0])
    if progress:
        print(f"  compile {time.time() - t0:.1f}s", flush=True)

    returns: list[float] = []
    for i, seed in enumerate(seeds):
        t1 = time.time()
        r = float(jax.device_get(run_episode(jax.random.PRNGKey(int(seed)))[0]))
        returns.append(r)
        if progress:
            print(
                f"  [{i + 1}/{len(seeds)}] seed={seed} return={r:.1f} "
                f"({time.time() - t1:.1f}s)",
                flush=True,
            )

    arr = np.asarray(returns, dtype=np.float64)
    train_seed = int(config.get("SEED", -1))
    summary: dict[str, Any] = {
        "protocol": "paper_greedy8",
        "train_seed": train_seed,
        "model": str(resolved),
        "env_name": str(env_name).lower(),
        "eval_seeds": seeds,
        "max_frames": int(max_frames),
        "eps": 0.0,
        "use_native_downscaling": bool(config.get("USE_NATIVE_DOWNSCALING", False)),
        "returns": returns,
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std(ddof=0)) if arr.size else float("nan"),
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "paper_seaquest_pqn_ref": PAPER_SEAQUEST_PQN_REF,
    }
    if progress:
        print(
            f"  → mean={summary['mean']:.1f} ± {summary['std']:.1f} "
            f"(paper PQN Seaquest ≈{PAPER_SEAQUEST_PQN_REF:.0f})",
            flush=True,
        )

    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if progress:
            print(f"  wrote {out}", flush=True)
    return summary


def publish_paper_greedy8_to_wandb(
    summary: dict[str, Any],
    *,
    ckpt_dir: Path | None,
    run_name: str,
    wandb_mode: str | None,
    tags: list[str] | None = None,
    entity: str = "paul-seitz-tu-darmstadt",
    project: str = "concept_based_learning",
) -> None:
    """Log paper greedy-8 as final metrics (resume train run if meta exists)."""
    import wandb

    from concept_based_learning.paper_eval.wandb_run_meta import load_wandb_run_meta

    mode = wandb_mode or "online"
    meta = load_wandb_run_meta(ckpt_dir) if ckpt_dir is not None else None
    init_kwargs: dict[str, Any] = {
        "entity": (meta or {}).get("entity") or entity,
        "project": (meta or {}).get("project") or project,
        "mode": mode,
    }
    if meta and meta.get("id"):
        init_kwargs["id"] = meta["id"]
        init_kwargs["resume"] = "allow"
    else:
        init_kwargs["name"] = f"{run_name}_paper_greedy8"
        init_kwargs["tags"] = list(tags or []) + ["phase:paper_greedy8"]

    run = wandb.init(**init_kwargs)
    try:
        metrics = {
            "paper_greedy8/mean": summary["mean"],
            "paper_greedy8/std": summary["std"],
            "paper_greedy8/min": summary["min"],
            "paper_greedy8/max": summary["max"],
            "paper_greedy8/median": summary["median"],
            "paper_greedy8/max_frames": summary["max_frames"],
            "paper_greedy8/n_seeds": len(summary.get("eval_seeds") or []),
            "paper_greedy8/paper_seaquest_ref": summary.get(
                "paper_seaquest_pqn_ref", PAPER_SEAQUEST_PQN_REF
            ),
        }
        for i, (seed, ret) in enumerate(
            zip(summary.get("eval_seeds") or [], summary.get("returns") or [])
        ):
            metrics[f"paper_greedy8/seed{seed}_return"] = float(ret)
            metrics[f"paper_greedy8/return_{i}"] = float(ret)
        wandb.summary.update(metrics)
        wandb.log(metrics)
        if ckpt_dir is not None:
            txt = ckpt_dir / "paper_greedy8_summary.txt"
            lines = [
                "paper_greedy8 summary",
                f"mean: {summary['mean']:.2f}",
                f"std: {summary['std']:.2f}",
                f"median: {summary['median']:.2f}",
                f"min: {summary['min']:.2f}",
                f"max: {summary['max']:.2f}",
                f"eval_seeds: {summary.get('eval_seeds')}",
                f"returns: {summary.get('returns')}",
                f"max_frames: {summary.get('max_frames')}",
                f"paper_ref: {summary.get('paper_seaquest_pqn_ref')}",
                "",
            ]
            txt.write_text("\n".join(lines), encoding="utf-8")
    finally:
        wandb.finish()
