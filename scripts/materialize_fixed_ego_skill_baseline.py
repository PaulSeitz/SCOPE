#!/usr/bin/env python3
"""Materialize same-scaffolding fixed ego-rule baselines for clip10 skillmech.

Keeps heuristic / labels / facing-frame / POV mapper identical to the learned
transfer stems; replaces only the skill Q-heads with hand-coded ego rules.

Arms:
  fixed_ego_rules — angle-servo COLLECT/AVOID/DESTROY
  constant_ego    — FORWARD / BACKWARD / FIRE constants

    python scripts/materialize_fixed_ego_skill_baseline.py
    python scripts/run_paper_eval.py --suite fixed_ego_skill_baseline \\
        --gpus 0 --lean
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SUITE = "fixed_ego_skill_baseline"
ARCH = (
    "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_"
    "noblocking_noweapon_surfacesub5_skillmech"
)
MODES = ("fixed_ego_rules", "constant_ego")
TRANSFER_ENVS = ("pong", "beamrider", "phoenix")


def _src_transfer(env: str) -> Path:
    tok = ARCH.removeprefix("seaquest_skills_ego_pov_")
    return _RQ2 / f"{env}_transfer_ego_pov_{tok}.yaml"


def _dst_transfer(env: str, mode: str) -> Path:
    tok = ARCH.removeprefix("seaquest_skills_ego_pov_")
    return _RQ2 / f"{env}_transfer_ego_pov_{tok}_{mode}.yaml"


def _src_sq_eval() -> Path:
    return _RQ1 / f"{ARCH}_eval_full.yaml"


def _dst_sq_eval(mode: str) -> Path:
    return _RQ1 / f"{ARCH}_{mode}_eval_full.yaml"


def _drop_ckpt_deps(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    skip_requires_list = False
    for line in lines:
        if line.startswith("PAPER_EVAL_SKILLS_CKPT:"):
            continue
        if line.startswith("PAPER_EVAL_REQUIRES_RQ1:"):
            skip_requires_list = True
            continue
        if skip_requires_list:
            if line.startswith("- "):
                continue
            skip_requires_list = False
        out.append(line)
    return "".join(out)


def _rewrite(text: str, *, mode: str, env: str | None) -> str:
    out = _drop_ckpt_deps(text)
    out = out.replace("PAPER_EVAL_KIND: transfer", "PAPER_EVAL_KIND: fixed_skills")
    out = out.replace("PAPER_EVAL_KIND: skills", "PAPER_EVAL_KIND: fixed_skills")
    out = out.replace("suite:ego_min_surfacesub5_skillmech_rw", f"suite:{SUITE}")

    # Base score only — mod tables are redundant for this attribution control.
    if "PAPER_EVAL_MOD_KEYS:" in out:
        out = re.sub(
            r"PAPER_EVAL_MOD_KEYS:\n(?:- .*\n)*",
            "PAPER_EVAL_MOD_KEYS:\n- base\n",
            out,
            count=1,
        )
    else:
        out = re.sub(
            r"(PAPER_EVAL_MODS:\s*\S+\n)",
            r"\1PAPER_EVAL_MOD_KEYS:\n- base\n",
            out,
            count=1,
        )

    if f"- baseline:{mode}" not in out:
        if "- phase:zero_shot\n" in out:
            out = out.replace(
                "- phase:zero_shot\n",
                f"- phase:zero_shot\n- baseline:{mode}\n",
            )
        elif "- ablation:skillmech\n" in out:
            out = out.replace(
                "- ablation:skillmech\n",
                f"- ablation:skillmech\n- baseline:{mode}\n",
            )
        else:
            out = out.replace(
                "WANDB_TAGS:\n",
                f"WANDB_TAGS:\n- baseline:{mode}\n",
            )

    if "SKILL_POLICY_MODE:" in out:
        out = re.sub(
            r"SKILL_POLICY_MODE:\s*\S+",
            f"SKILL_POLICY_MODE: {mode}",
            out,
        )
    else:
        out = out.replace(
            "SKILL_INPUT_MODE: ego\n",
            (
                "SKILL_INPUT_MODE: ego\n"
                f"SKILL_POLICY_MODE: {mode}\n"
                "FIXED_EGO_ALIGN_DEG: 22.5\n"
            ),
            1,
        )

    suffix = f"_{mode}"
    if env is not None:
        tok = ARCH.removeprefix("seaquest_skills_ego_pov_")
        old_save = f"./outputs/models/paper_eval/rq2/{env}_transfer_ego_pov_{tok}"
        out = out.replace(old_save, old_save + suffix)
    # Tag wandb run names that end with the skillmech stem (avoid ARCHITECTURE_OF).
    out = re.sub(
        r"(WANDB_RUN_NAME:.*skillmech)(\n)",
        rf"\1{suffix}\2",
        out,
    )
    # Keep architecture pin on the learned train stem.
    out = out.replace(
        f"PAPER_EVAL_ARCHITECTURE_OF: {ARCH}{suffix}",
        f"PAPER_EVAL_ARCHITECTURE_OF: {ARCH}",
    )
    return out


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def materialize(*, dry: bool) -> None:
    sq_src = _src_sq_eval()
    if not sq_src.exists():
        raise FileNotFoundError(sq_src)
    for mode in MODES:
        body = _rewrite(sq_src.read_text(encoding="utf-8"), mode=mode, env=None)
        _write(_dst_sq_eval(mode), body, dry=dry)

    for env in TRANSFER_ENVS:
        src = _src_transfer(env)
        if not src.exists():
            raise FileNotFoundError(src)
        for mode in MODES:
            body = _rewrite(src.read_text(encoding="utf-8"), mode=mode, env=env)
            _write(_dst_transfer(env, mode), body, dry=dry)


def stem_lists() -> list[str]:
    stems: list[str] = []
    for mode in MODES:
        stems.append(f"{ARCH}_{mode}_eval_full")
        tok = ARCH.removeprefix("seaquest_skills_ego_pov_")
        for env in TRANSFER_ENVS:
            stems.append(f"{env}_transfer_ego_pov_{tok}_{mode}")
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--print-stems", action="store_true")
    args = parser.parse_args()
    if args.print_stems:
        for s in stem_lists():
            print(s)
        return
    materialize(dry=args.dry)
    print(f"\n{len(stem_lists())} stems ({SUITE})")


if __name__ == "__main__":
    main()
