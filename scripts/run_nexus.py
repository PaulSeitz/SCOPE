#!/usr/bin/env python3
"""Train / mod-eval NEXUS (symbolic_options) against CBL Seaquest + mods.

Default jaxatari is ``JAXATARI_ROOT`` / the installed robustness fork (cluster).
Pass ``--vendored-seaquest`` only to force NEXUS's bundled old Seaquest.

Cluster setup
-------------
  git submodule update --init --recursive third_party/symbolic_options
  export NEXUS_ROOT=$PWD/third_party/symbolic_options   # or any clone path
  export JAXATARI_ROOT=/path/to/JAXAtari-seaquest-robustness
  # optional: cd $NEXUS_ROOT && uv sync

Examples
--------
  python scripts/run_nexus.py train --alg cbl_seaquest_nexus_symbolic_paper
  python scripts/run_nexus.py train --alg cbl_seaquest_hrl_baseline_paper

  # Full paper_eval seaquest mod allowlist via jaxatari.make:
  python scripts/run_nexus.py mod-eval --alg cbl_seaquest_nexus_symbolic_paper \\
      --ckpt outputs/models/nexus/cbl_seaquest_nexus_symbolic_paper

  python scripts/run_nexus.py train --alg cbl_seaquest_nexus_symbolic_paper --trial
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from nexus.paths import (  # noqa: E402
    nexus_alg_config_dir,
    resolve_jaxatari_root,
    resolve_nexus_root,
)

CBL_ALG_DIR = PROJECT_ROOT / "configs" / "experiment" / "nexus" / "alg"
DEFAULT_SAVE_ROOT = PROJECT_ROOT / "outputs" / "models" / "nexus"


def _sync_alg_configs(nexus_root: Path) -> None:
    dest = nexus_alg_config_dir(nexus_root)
    dest.mkdir(parents=True, exist_ok=True)
    for src in sorted(CBL_ALG_DIR.glob("cbl_*.yaml")):
        target = dest / src.name
        shutil.copy2(src, target)
        print(f"Synced alg config → {target.relative_to(nexus_root)}")


def _hydra_overrides(args: argparse.Namespace, save_path: Path) -> list[str]:
    overs = [
        f"SAVE_PATH={save_path}",
        f"SEED={args.seed}",
        f"NUM_SEEDS={args.num_seeds}",
        f"PROJECT={args.wandb_project}",
        f"WANDB_MODE={args.wandb_mode}",
    ]
    if args.trial:
        overs += [
            "alg.TOTAL_TIMESTEPS=200000",
            "alg.TOTAL_TIMESTEPS_DECAY=200000",
            "alg.NUM_ENVS=8",
            "alg.NUM_STEPS=32",
            "alg.NUM_MINIBATCHES=4",
            "alg.TEST_NUM_ENVS=8",
            "alg.TEST_NUM_STEPS=200",
        ]
    if args.extra:
        overs.extend(args.extra)
    return overs


def _train(args: argparse.Namespace) -> int:
    nexus_root = resolve_nexus_root(args.nexus_root)
    jaxatari_root = resolve_jaxatari_root(args.jaxatari_root)
    _sync_alg_configs(nexus_root)

    save_path = Path(args.save_path) if args.save_path else DEFAULT_SAVE_ROOT / args.alg
    save_path = save_path.resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    use_vendored = bool(args.vendored_seaquest)
    runner = nexus_root / "_cbl_nexus_runner.py"
    runner.write_text(
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(PROJECT_ROOT / 'scripts')!r})",
                "from nexus.compat import prepare_nexus_runtime",
                "prepare_nexus_runtime(",
                f"    nexus_root=Path({str(nexus_root)!r}),",
                f"    jaxatari_root=Path({str(jaxatari_root)!r}),",
                f"    use_vendored_seaquest={use_vendored},",
                ")",
                "sys.argv = "
                + repr(["main.py", f"+alg={args.alg}"] + _hydra_overrides(args, save_path)),
                "import runpy",
                f"runpy.run_path({str(nexus_root / 'main.py')!r}, run_name='__main__')",
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["NEXUS_ROOT"] = str(nexus_root)
    env["JAXATARI_ROOT"] = str(jaxatari_root)
    env["PAPER_EVAL_JAXATARI_ROOT"] = str(jaxatari_root)
    # Default: installed/cluster jaxatari first; NEXUS src for symbolic_options.
    # --vendored-seaquest: NEXUS bundled JAXAtari instead.
    jax_src = (
        str(nexus_root / "JAXAtari" / "src")
        if use_vendored
        else str(Path(jaxatari_root) / "src")
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [
            jax_src,
            str(nexus_root / "src"),
            str(PROJECT_ROOT / "scripts"),
            env.get("PYTHONPATH", ""),
        ]
    )

    cmd = [sys.executable, str(runner)]
    print("Running:", " ".join(cmd))
    print(f"  NEXUS_ROOT={nexus_root}")
    print(f"  JAXATARI_ROOT={jaxatari_root}")
    print(f"  vendored_seaquest={use_vendored}")
    print(f"  SAVE_PATH={save_path}")
    try:
        return subprocess.call(cmd, cwd=str(nexus_root), env=env)
    finally:
        if runner.is_file() and not args.keep_bootstrap:
            runner.unlink(missing_ok=True)


def _mod_eval(args: argparse.Namespace) -> int:
    from nexus.mod_eval import run_nexus_mod_eval

    nexus_root = resolve_nexus_root(args.nexus_root)
    jaxatari_root = resolve_jaxatari_root(args.jaxatari_root)
    _sync_alg_configs(nexus_root)

    ckpt_dir = Path(args.ckpt) if args.ckpt else DEFAULT_SAVE_ROOT / args.alg
    out_json = (
        Path(args.out_json)
        if args.out_json
        else PROJECT_ROOT
        / "outputs"
        / "models"
        / "paper_eval"
        / "eval"
        / f"{args.alg}_mods.json"
    )
    run_nexus_mod_eval(
        nexus_root=nexus_root,
        jaxatari_root=jaxatari_root,
        alg=args.alg,
        ckpt_dir=ckpt_dir,
        seed=args.seed,
        num_episodes=args.eval_episodes,
        num_seeds=getattr(args, "eval_num_seeds", 1),
        max_frames=args.eval_max_frames,
        trial=args.trial,
        mod_keys=args.mod_keys,
        output_json=out_json,
        use_vendored_seaquest=bool(args.vendored_seaquest),
    )
    return 0


def _list_algs(_: argparse.Namespace) -> int:
    print("CBL NEXUS alg overlays:")
    for p in sorted(CBL_ALG_DIR.glob("cbl_*.yaml")):
        print(f"  {p.stem}")
    print("\nUpstream NEXUS Seaquest algs (also usable via --alg):")
    for name in (
        "pqn_jaxtari_sea1",
        "pqn_jaxtari_sea2_hier_baseline",
        "pqn_jaxtari_sea2_hier",
        "pqn_jaxtari_sea3_hier_llm",
        "pqn_jaxtari_sea4_hier_comb",
    ):
        print(f"  {name}")
    return 0


def _add_root_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--nexus-root",
        default=None,
        help="symbolic_options path (default: NEXUS_ROOT or third_party/symbolic_options)",
    )
    p.add_argument(
        "--jaxatari-root",
        default=None,
        help="JAXAtari-seaquest-robustness path (default: JAXATARI_ROOT / PAPER_EVAL_JAXATARI_ROOT)",
    )
    p.add_argument(
        "--vendored-seaquest",
        action="store_true",
        help="Use NEXUS bundled (old) Seaquest instead of JAXATARI_ROOT / installed fork (default)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-algs")
    _add_root_flags(p_list)
    p_list.set_defaults(func=_list_algs)

    p_train = sub.add_parser("train")
    _add_root_flags(p_train)
    p_train.add_argument("--alg", required=True)
    p_train.add_argument("--save-path", default=None)
    p_train.add_argument("--seed", type=int, default=44)
    p_train.add_argument("--num-seeds", type=int, default=1)
    p_train.add_argument("--wandb-mode", default="online")
    p_train.add_argument("--wandb-project", default="concept_based_learning_nexus")
    p_train.add_argument("--trial", action="store_true")
    p_train.add_argument("--keep-bootstrap", action="store_true")
    p_train.add_argument("--extra", nargs="*", default=[])
    p_train.set_defaults(func=_train)

    p_eval = sub.add_parser("mod-eval")
    _add_root_flags(p_eval)
    p_eval.add_argument("--alg", required=True)
    p_eval.add_argument("--ckpt", default=None)
    p_eval.add_argument("--seed", type=int, default=10044)
    p_eval.add_argument("--eval-episodes", type=int, default=3)
    p_eval.add_argument("--eval-num-seeds", type=int, default=1)
    p_eval.add_argument("--eval-max-frames", type=int, default=27000)
    p_eval.add_argument("--trial", action="store_true")
    p_eval.add_argument("--mod-keys", nargs="*", default=None)
    p_eval.add_argument("--out-json", default=None)
    p_eval.set_defaults(func=_mod_eval)

    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
