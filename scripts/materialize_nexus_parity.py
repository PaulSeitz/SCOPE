#!/usr/bin/env python3
"""Materialize sanity arms matching the NEXUS mainline *outer* schedule.

NEXUS Seaquest claim (``pqn_jaxtari_sea3_hier_llm`` / paper hier Table-1):
  TOTAL_TIMESTEPS = 5e7, NUM_ENVS = 1024, NUM_STEPS = 128
  → NUM_UPDATES = floor(5e7 / (1024×128)) = 381
  → batch-aligned total = 381 × 131072 = 49_938_432

Keeps **our locked learning HPs** (skills lr1e3 + ε B; tandem e01); only the
outer loop matches NEXUS.

Hard caps: none on ``NUM_ENVS`` / ``TOTAL_TIMESTEPS`` in CBL trainers.
Soft constraint: ``(NUM_STEPS * NUM_ENVS) % NUM_MINIBATCHES == 0`` (OK at 64).
Soft risk: tandem pixel meta @ 1024 envs may OOM (large rollout buffer).

    python scripts/materialize_nexus_parity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import materialize_final_10m as f10

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "configs/experiment/paper_eval/nexus_parity"

SUITE = "nexus_parity"
BUDGET_TAG = "50m_nexus"
TIMESTEPS = 49_938_432  # 381 × 1024 × 128
NUM_ENVS = 1024
NUM_STEPS = 128
NUM_UPDATES = TIMESTEPS // (NUM_ENVS * NUM_STEPS)
OUT_MODELS = "./outputs/models/paper_eval/nexus_parity"

STEM_SKILLS = "nexus_parity_skills_ego"
STEM_TANDEM = "nexus_parity_tandem_pixel_ego"


def nexus_parity_stems() -> tuple[str, ...]:
    """Tandem first (OOM-fast), then skills, then home evals."""
    return (
        STEM_TANDEM,
        STEM_SKILLS,
        f"{STEM_TANDEM}_eval",
        f"{STEM_SKILLS}_eval",
    )


def _rewrite(
    body: str,
    *,
    stem: str,
    header: str,
    run_name: str,
) -> str:
    """Take a final_10m YAML and retarget paths / outer-loop / suite tags."""
    replacements = [
        (f"SAVE_PATH: {f10.OUT_MODELS}/{f10.STEM_SKILLS}", f"SAVE_PATH: {OUT_MODELS}/{stem}"),
        (
            f"SAVE_PATH: {f10.OUT_MODELS}/{f10.STEM_TANDEM_PIXEL}",
            f"SAVE_PATH: {OUT_MODELS}/{stem}",
        ),
        (
            f"{f10.OUT_MODELS}/{f10.STEM_SKILLS}/",
            f"{OUT_MODELS}/{stem}/",
        ),
        (
            f"{f10.OUT_MODELS}/{f10.STEM_TANDEM_PIXEL}/",
            f"{OUT_MODELS}/{stem}/",
        ),
        (f"- {f10.STEM_SKILLS}", f"- {stem}"),
        (f"- {f10.STEM_TANDEM_PIXEL}", f"- {stem}"),
        (f"PAPER_EVAL_ARCHITECTURE_OF: {f10.STEM_SKILLS}", f"PAPER_EVAL_ARCHITECTURE_OF: {stem}"),
        (
            f"PAPER_EVAL_ARCHITECTURE_OF: {f10.STEM_TANDEM_PIXEL}",
            f"PAPER_EVAL_ARCHITECTURE_OF: {stem}",
        ),
        (f"- suite:{f10.SUITE}", f"- suite:{SUITE}"),
        (f"- budget:{f10.BUDGET_TAG}", f"- budget:{BUDGET_TAG}"),
        ("  TOTAL_TIMESTEPS: 9994240", f"  TOTAL_TIMESTEPS: {TIMESTEPS}"),
        ("  NUM_ENVS: 128", f"  NUM_ENVS: {NUM_ENVS}"),
        ("  NUM_STEPS: 128", f"  NUM_STEPS: {NUM_STEPS}"),
    ]
    # Header + run name: replace first comment line and WANDB_RUN_NAME
    lines = body.splitlines()
    if lines and lines[0].startswith("#"):
        lines[0] = header
    for i, line in enumerate(lines):
        if line.startswith("WANDB_RUN_NAME:"):
            lines[i] = f"WANDB_RUN_NAME: {run_name}"
            break
    body = "\n".join(lines) + "\n"
    for old, new in replacements:
        body = body.replace(old, new)
    # Inject parity tags after budget tag (once).
    needle = f"- budget:{BUDGET_TAG}\n"
    insert = (
        f"- budget:{BUDGET_TAG}\n"
        f"- parity:nexus_outer\n"
        f"- num_envs:{NUM_ENVS}\n"
        f"- num_updates:{NUM_UPDATES}\n"
    )
    if needle in body and insert not in body:
        body = body.replace(needle, insert, 1)
    return body


def _manifest() -> str:
    return f10._lines(
        [
            "# nexus_parity inventory (not runnable).",
            f"suite: {SUITE}",
            f"budget_tag: {BUDGET_TAG}",
            f"total_timesteps: {TIMESTEPS}",
            f"num_envs: {NUM_ENVS}",
            f"num_steps: {NUM_STEPS}",
            f"num_updates: {NUM_UPDATES}",
            "source: floor(5e7/(1024*128))*1024*128 == NEXUS sea3 NUM_UPDATES",
            "keeps: CBL locked skill/tandem learning HPs",
            f"skills_train: {STEM_SKILLS}",
            f"skills_eval: {STEM_SKILLS}_eval",
            f"tandem_train: {STEM_TANDEM}",
            f"tandem_eval: {STEM_TANDEM}_eval",
            f"outputs_root: {OUT_MODELS}",
            "hard_caps: none on NUM_ENVS/TOTAL_TIMESTEPS",
            "constraint: (NUM_STEPS*NUM_ENVS) % NUM_MINIBATCHES == 0",
            "compare: final_10m @128 envs has 610 updates (> 381 here)",
        ]
    )


def materialize(*, dry: bool) -> None:
    print(
        f"nexus_parity → {_OUT.relative_to(_ROOT)}  "
        f"timesteps={TIMESTEPS} envs={NUM_ENVS} updates={NUM_UPDATES}"
    )
    pairs = [
        (
            _OUT / f"{STEM_SKILLS}.yaml",
            _rewrite(
                f10._skills_train(),
                stem=STEM_SKILLS,
                header=(
                    f"# Seaquest ego skills — NEXUS-parity outer loop "
                    f"({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}={TIMESTEPS}); "
                    f"locked lr1e3 + ε B."
                ),
                run_name=f"sq_{SUITE}_skills_ego_{BUDGET_TAG}",
            ),
        ),
        (
            _OUT / f"{STEM_SKILLS}_eval.yaml",
            _rewrite(
                f10._skills_eval(),
                stem=STEM_SKILLS,
                header="# Home mod eval — nexus_parity skills.",
                run_name=f"sq_{SUITE}_eval_skills_ego_{BUDGET_TAG}",
            ),
        ),
        (
            _OUT / f"{STEM_TANDEM}.yaml",
            _rewrite(
                f10._tandem_train(
                    stem=f10.STEM_TANDEM_PIXEL,
                    meta_mode="pixel_embed",
                    meta_tag="pixel",
                ),
                stem=STEM_TANDEM,
                header=(
                    f"# Seaquest tandem pixel — NEXUS-parity outer loop "
                    f"({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}={TIMESTEPS}); "
                    f"locked e01. Soft risk: pixel meta @ {NUM_ENVS} envs may OOM."
                ),
                run_name=f"sq_{SUITE}_tandem_pixel_{BUDGET_TAG}",
            ),
        ),
        (
            _OUT / f"{STEM_TANDEM}_eval.yaml",
            _rewrite(
                f10._tandem_eval(
                    stem=f10.STEM_TANDEM_PIXEL,
                    meta_mode="pixel_embed",
                    meta_tag="pixel",
                ),
                stem=STEM_TANDEM,
                header="# Home mod eval — nexus_parity tandem pixel.",
                run_name=f"sq_{SUITE}_eval_tandem_pixel_{BUDGET_TAG}",
            ),
        ),
        (_OUT / "manifest.yaml", _manifest()),
    ]
    for path, body in pairs:
        f10._write(path, body, dry=dry)
    print(
        "\nSanity launch:\n"
        f"  python scripts/run_paper_eval.py --only {STEM_SKILLS} "
        f"--gpus 0 --skip-rq-gate --lean\n"
        f"  python scripts/run_paper_eval.py --only {STEM_TANDEM} "
        f"--gpus 0 --skip-rq-gate --lean"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    materialize(dry=bool(args.dry_run))


if __name__ == "__main__":
    main()
