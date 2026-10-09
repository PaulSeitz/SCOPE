#!/usr/bin/env python3
"""Cycle paper_core mods in the hierarchical PQN visualizer (Esc/q → next mod).

Uses one live visualizer process per mod. Esc / q quit that process and advance;
after the last mod the script exits. Prefer ``PYENV_VERSION=jaxtari-fork`` so
Seaquest mods come from ``Documents/JAXAtari-seaquest-robustness``.

Example::

    PYENV_VERSION=jaxtari-fork PYTHONUNBUFFERED=1 \\
      python -u scripts/visualize_skills_mod_tour.py \\
      --model tmp/skills_ib_ego_vs_pixel/ckpts/pixel/hierarchical_pqn_seaquest_seed45.safetensors \\
      --paper-config configs/experiment/paper_eval/optim_scale/skills_ib_pixel_outline_heu_v2_seed45_eval.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_mod_keys(paper_config: Path) -> list[str]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(paper_config)
    keys = list(cfg.get("PAPER_EVAL_MOD_KEYS") or [])
    if keys:
        return [str(k) for k in keys]
    # Fallback: seaquest paper_core allowlist
    from concept_based_learning.paper_eval.mods import load_mod_allowlist

    raw = load_mod_allowlist(str(cfg.get("ENV_NAME") or "seaquest"))
    keys = list(raw.get("paper_core") or [])
    if keys:
        return [str(k) for k in keys]
    return ["base"]


def _warn_if_wrong_jaxatari() -> None:
    try:
        import jaxtari
    except ImportError as e:
        print(f"WARNING: jaxtari import failed: {e}", file=sys.stderr)
        return
    path = Path(jaxtari.__file__).resolve()
    print(f"jaxtari: {path}")
    if "seaquest-robustness" not in str(path):
        print(
            "WARNING: jaxtari is not JAXAtari-seaquest-robustness — mods may be "
            "missing/stale. Use: PYENV_VERSION=jaxtari-fork",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--paper-config",
        type=str,
        required=True,
        help="Eval YAML providing PAPER_EVAL_MOD_KEYS (+ architecture merge).",
    )
    parser.add_argument(
        "--mods",
        nargs="*",
        default=None,
        help="Override mod key list (default: PAPER_EVAL_MOD_KEYS from paper-config).",
    )
    parser.add_argument(
        "--start-at",
        type=str,
        default=None,
        help="Skip ahead to this mod key (inclusive).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args after '--' forwarded to pqn_visualizer.",
    )
    args = parser.parse_args()

    model = Path(args.model)
    if not model.is_file():
        raise SystemExit(f"Model not found: {model}")
    paper_config = Path(args.paper_config)
    if not paper_config.is_file():
        # allow repo-relative
        alt = _ROOT / args.paper_config
        paper_config = alt if alt.is_file() else paper_config
    if not paper_config.is_file():
        raise SystemExit(f"Paper config not found: {args.paper_config}")

    _warn_if_wrong_jaxatari()

    mod_keys = list(args.mods) if args.mods is not None else _load_mod_keys(paper_config)
    if args.start_at:
        if args.start_at not in mod_keys:
            raise SystemExit(
                f"--start-at={args.start_at!r} not in mod list: {mod_keys}"
            )
        mod_keys = mod_keys[mod_keys.index(args.start_at) :]

    # Prefer module entry so PYTHONPATH/src layout matches training.
    viz_cmd_base = [
        sys.executable,
        "-u",
        "-m",
        "concept_based_learning.pqn_visualizer",
        "--model",
        str(model.resolve()),
        "--paper-config",
        str(paper_config.resolve()),
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
        "--wandb-mode",
        "disabled",
        "--display",
        "pygame",
    ]
    extra = list(args.extra or [])
    if extra and extra[0] == "--":
        extra = extra[1:]

    env = os.environ.copy()
    # Ensure src/ is importable even if invoked oddly.
    src = str(_ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    n = len(mod_keys)
    print(
        f"Mod tour ({n}): {', '.join(mod_keys)}\n"
        "Esc / q → next mod | close window → next mod | Ctrl-C → abort tour"
    )

    for i, key in enumerate(mod_keys, start=1):
        print("\n" + "=" * 72)
        print(f"[{i}/{n}] MOD = {key!r}   (Esc/q advances)")
        print("=" * 72)
        cmd = list(viz_cmd_base)
        # base = no GAME_MODS; others pass the single mod name.
        if key == "base":
            cmd += ["--game-mods"]
        else:
            cmd += ["--game-mods", key]
        cmd += extra
        print("exec:", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, cwd=str(_ROOT), env=env)
        except KeyboardInterrupt:
            print("\nTour aborted.")
            raise SystemExit(130)
        if proc.returncode not in (0, None):
            # Visualizer usually exits 0 on q; non-zero may be a crash — stop.
            if proc.returncode != 0:
                print(
                    f"Visualizer exited with code {proc.returncode}; "
                    "stopping tour (re-run with --start-at to resume).",
                    file=sys.stderr,
                )
                raise SystemExit(proc.returncode)

    print("\nAll mods done.")


if __name__ == "__main__":
    main()
