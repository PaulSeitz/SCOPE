#!/usr/bin/env python3
"""Materialize Seaquest ego-skills + tandem-pixel at PQN optimal-scaling outer loop.

Reference (flat PQN pixel)::

    /home/paul/Downloads/pqn_jaxatari_pixel_optimal_scaling.yaml
    → TOTAL_TIMESTEPS≈5e7, NUM_ENVS=1024, NUM_STEPS=32, NUM_EPOCHS=2,
      NUM_MINIBATCHES=256, LR=2.5e-4 (+ linear decay), MAX_GRAD_NORM=10, …

CBL mapping
-----------
* Outer loop: 1024 envs × 50M agent-steps (batch-aligned).
  Hierarchical arms keep ``NUM_STEPS=128`` (CBL / NEXUS-parity convention)
  → ``NUM_UPDATES = floor(5e7 / (1024×128)) = 381``
  → ``TOTAL_TIMESTEPS = 381 × 131072 = 49_938_432``.
* Skill / tandem *learning* HPs stay CBL locks (ε B skills, tandem e01 meta),
  except ``SKILL_LR=5e-4`` (smoke recipe; 10M ``lr1e3`` and ``lr5e4`` both
  collapsed on Seaquest — longer 50M budget + 5e-4 is the working scale).
* ``NUM_EPOCHS=2`` / ``NUM_MINIBATCHES=256`` taken from the optimal-scaling
  YAML (batch constraint satisfied: ``(128×1024) % 256 == 0``).
* Flat PQN ``LR=2.5e-4`` / ε 1→0.001 do **not** override skill/meta LRs.
* 3 train seeds via separate stems (44/45/46) — hierarchical ``NUM_SEEDS>1``
  only saves seed-0 under the base ``SEED`` filename.

    python scripts/materialize_optim_scale.py
    python scripts/materialize_optim_scale.py --dry-run

    CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \\
      --suite optim_scale_sq --wait-deps --lean --skip-rq-gate
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import materialize_final_10m as f10

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "configs/experiment/paper_eval/optim_scale"
_REF_SRC = Path("/home/paul/Downloads/pqn_jaxatari_pixel_optimal_scaling.yaml")

SUITE = "optim_scale_sq"
BUDGET_TAG = "50m_optim"
TIMESTEPS = 49_938_432  # 381 × 1024 × 128
NUM_ENVS = 1024
NUM_STEPS = 128
NUM_UPDATES = TIMESTEPS // (NUM_ENVS * NUM_STEPS)
NUM_EPOCHS = 2
NUM_MINIBATCHES = 256
OUT_MODELS = "./outputs/models/paper_eval/optim_scale"

# Smoke / longest healthy Seaquest skills recipe (noclip_sparse_smoke @ 20M).
SKILL_LR = 0.0005
SEEDS = (44, 45, 46)

MOD_KEYS = f10.MOD_KEYS


def skills_stem(seed: int) -> str:
    return f"optim_scale_skills_ego_seed{seed}"


def tandem_stem(seed: int) -> str:
    return f"optim_scale_tandem_pixel_ego_seed{seed}"


def optim_scale_sq_stems() -> tuple[str, ...]:
    """Tandem trains first (OOM-fast), then skills, then home evals."""
    tandem_trains = tuple(tandem_stem(s) for s in SEEDS)
    skills_trains = tuple(skills_stem(s) for s in SEEDS)
    tandem_evals = tuple(f"{s}_eval" for s in tandem_trains)
    skills_evals = tuple(f"{s}_eval" for s in skills_trains)
    return tandem_trains + skills_trains + tandem_evals + skills_evals


def _patch_outer_and_lr(body: str, *, skill_lr: float = SKILL_LR) -> str:
    """Retarget final_10m body → optim_scale outer loop + SKILL_LR."""
    reps = [
        ("  TOTAL_TIMESTEPS: 9994240", f"  TOTAL_TIMESTEPS: {TIMESTEPS}"),
        ("  NUM_ENVS: 128", f"  NUM_ENVS: {NUM_ENVS}"),
        ("  NUM_STEPS: 128", f"  NUM_STEPS: {NUM_STEPS}"),
        ("  NUM_EPOCHS: 4", f"  NUM_EPOCHS: {NUM_EPOCHS}"),
        ("  NUM_MINIBATCHES: 64", f"  NUM_MINIBATCHES: {NUM_MINIBATCHES}"),
        ("  SKILL_LR: 0.001", f"  SKILL_LR: {skill_lr}"),
        ("  SKILL_LR: 0.0005", f"  SKILL_LR: {skill_lr}"),
        ("- budget:10m", f"- budget:{BUDGET_TAG}"),
        ("- suite:final_10m", f"- suite:{SUITE}"),
        ("- lock:skills_lr1e3", "- lock:skills_lr5e4"),
        ("- lock:skills_lr5e4", "- lock:skills_lr5e4"),  # idempotent
        ("- lock:tandem_e01", "- lock:tandem_e01"),
        (
            f"SAVE_PATH: {f10.OUT_MODELS}/{f10.STEM_SKILLS}",
            "SAVE_PATH: PLACEHOLDER",
        ),
        (
            f"SAVE_PATH: {f10.OUT_MODELS}/{f10.STEM_TANDEM_PIXEL}",
            "SAVE_PATH: PLACEHOLDER",
        ),
    ]
    for old, new in reps:
        body = body.replace(old, new)
    # Inject scale tags once after budget tag.
    needle = f"- budget:{BUDGET_TAG}\n"
    insert = (
        f"- budget:{BUDGET_TAG}\n"
        f"- scale:pqn_optim\n"
        f"- num_envs:{NUM_ENVS}\n"
        f"- num_updates:{NUM_UPDATES}\n"
        f"- num_epochs:{NUM_EPOCHS}\n"
        f"- num_minibatches:{NUM_MINIBATCHES}\n"
    )
    if needle in body and "- scale:pqn_optim\n" not in body:
        body = body.replace(needle, insert, 1)
    return body


def _retarget(
    body: str,
    *,
    stem: str,
    seed: int,
    header: str,
    run_name: str,
    kind: str,
) -> str:
    body = _patch_outer_and_lr(body)
    lines = body.splitlines()
    if lines and lines[0].startswith("#"):
        lines[0] = header
    out: list[str] = []
    for line in lines:
        if line.startswith("WANDB_RUN_NAME:"):
            out.append(f"WANDB_RUN_NAME: {run_name}")
        elif line.startswith("SAVE_PATH:"):
            out.append(f"SAVE_PATH: {OUT_MODELS}/{stem}")
        elif line.startswith("SEED:"):
            out.append(f"SEED: {seed}")
        elif line.startswith("EVAL_SEED:"):
            out.append(f"EVAL_SEED: {seed + 10_000}")
        elif "PAPER_EVAL_REQUIRES_RQ1:" in line:
            out.append(line)
        elif line.strip().startswith("- final_10m_") or line.strip().startswith(
            "- optim_scale_"
        ):
            # require stem rewritten below
            out.append(line)
        elif line.startswith("PAPER_EVAL_ARCHITECTURE_OF:"):
            out.append(f"PAPER_EVAL_ARCHITECTURE_OF: {stem}")
        elif line.startswith("PAPER_EVAL_SKILLS_CKPT:"):
            if kind == "skills":
                ckpt = (
                    f"{OUT_MODELS}/{stem}/seaquest/"
                    f"hierarchical_pqn_seaquest_seed{seed}.safetensors"
                )
            else:
                ckpt = (
                    f"{OUT_MODELS}/{stem}/seaquest/"
                    f"hierarchical_pqn_seaquest_seed{seed}_skills.safetensors"
                )
            out.append(f"PAPER_EVAL_SKILLS_CKPT: {ckpt}")
        elif line.startswith("PAPER_EVAL_META_CKPT:"):
            ckpt = (
                f"{OUT_MODELS}/{stem}/seaquest/"
                f"hierarchical_pqn_seaquest_seed{seed}_meta.safetensors"
            )
            out.append(f"PAPER_EVAL_META_CKPT: {ckpt}")
        else:
            out.append(line)
    body = "\n".join(out) + "\n"
    # Fix requires list to this stem
    body = body.replace(
        f"- {f10.STEM_SKILLS}\n",
        f"- {stem}\n",
    )
    body = body.replace(
        f"- {f10.STEM_TANDEM_PIXEL}\n",
        f"- {stem}\n",
    )
    # Any leftover final_10m paths
    body = body.replace(f"{f10.OUT_MODELS}/", f"{OUT_MODELS}/")
    body = body.replace("PLACEHOLDER", f"{OUT_MODELS}/{stem}")
    # Ensure lock tag for skills
    if kind == "skills" and "lock:skills_lr5e4" not in body:
        body = body.replace("- lr:constant\n", "- lr:constant\n- lock:skills_lr5e4\n")
    return body


def _skills_train(seed: int) -> str:
    stem = skills_stem(seed)
    # Prefer already-patched local final_10m (may be lr5e4); materializer base is fine.
    base = f10._skills_train("seaquest")
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        header=(
            f"# Seaquest ego skills — PQN optimal-scaling outer "
            f"({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}={TIMESTEPS}); "
            f"SKILL_LR={SKILL_LR} + ε B; seed={seed}."
        ),
        run_name=f"sq_{SUITE}_skills_ego_seed{seed}_{BUDGET_TAG}",
        kind="skills",
    )


def _skills_eval(seed: int) -> str:
    stem = skills_stem(seed)
    base = f10._skills_eval("seaquest")
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        header=f"# Home mod eval — optim_scale skills seed{seed}.",
        run_name=f"sq_{SUITE}_eval_skills_ego_seed{seed}_{BUDGET_TAG}",
        kind="skills",
    )


def _tandem_train(seed: int) -> str:
    stem = tandem_stem(seed)
    base = f10._tandem_train(
        stem=f10.STEM_TANDEM_PIXEL,
        meta_mode="pixel_embed",
        meta_tag="pixel",
    )
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        header=(
            f"# Seaquest tandem pixel — PQN optimal-scaling outer "
            f"({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}={TIMESTEPS}); "
            f"e01 meta + SKILL_LR={SKILL_LR}; seed={seed}. "
            f"Soft risk: pixel meta @ {NUM_ENVS} envs may OOM."
        ),
        run_name=f"sq_{SUITE}_tandem_pixel_seed{seed}_{BUDGET_TAG}",
        kind="tandem",
    )


def _tandem_eval(seed: int) -> str:
    stem = tandem_stem(seed)
    base = f10._tandem_eval(
        stem=f10.STEM_TANDEM_PIXEL,
        meta_mode="pixel_embed",
        meta_tag="pixel",
    )
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        header=f"# Home mod eval — optim_scale tandem pixel seed{seed}.",
        run_name=f"sq_{SUITE}_eval_tandem_pixel_seed{seed}_{BUDGET_TAG}",
        kind="tandem",
    )


def _manifest() -> str:
    return f10._lines(
        [
            "# optim_scale_sq inventory (not runnable).",
            f"suite: {SUITE}",
            f"budget_tag: {BUDGET_TAG}",
            f"total_timesteps: {TIMESTEPS}",
            f"num_envs: {NUM_ENVS}",
            f"num_steps: {NUM_STEPS}",
            f"num_updates: {NUM_UPDATES}",
            f"num_epochs: {NUM_EPOCHS}",
            f"num_minibatches: {NUM_MINIBATCHES}",
            f"skill_lr: {SKILL_LR}",
            f"seeds: {list(SEEDS)}",
            "ref: pqn_jaxatari_pixel_optimal_scaling.yaml (outer scale)",
            "skills: ε B + SKILL_LR=5e-4 (smoke recipe; not 1e-3)",
            "tandem: e01 lock (META_LR=5e-4, switch=-0.05) + same SKILL_LR",
            f"outputs_root: {OUT_MODELS}",
            "stems:",
            *[f"  - {s}" for s in optim_scale_sq_stems()],
        ]
    )


def materialize(*, dry: bool) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"optim_scale_sq → {_OUT.relative_to(_ROOT)}  "
        f"timesteps={TIMESTEPS} envs={NUM_ENVS} updates={NUM_UPDATES} "
        f"SKILL_LR={SKILL_LR} seeds={SEEDS}"
    )
    pairs: list[tuple[Path, str]] = []
    for seed in SEEDS:
        pairs.append((_OUT / f"{skills_stem(seed)}.yaml", _skills_train(seed)))
        pairs.append((_OUT / f"{skills_stem(seed)}_eval.yaml", _skills_eval(seed)))
        pairs.append((_OUT / f"{tandem_stem(seed)}.yaml", _tandem_train(seed)))
        pairs.append((_OUT / f"{tandem_stem(seed)}_eval.yaml", _tandem_eval(seed)))
    pairs.append((_OUT / "manifest.yaml", _manifest()))

    for path, body in pairs:
        f10._write(path, body, dry=dry)

    # Keep a copy of the flat-PQN reference next to the suite.
    ref_dst = _OUT / "pqn_jaxatari_pixel_optimal_scaling.yaml"
    if _REF_SRC.is_file() and not dry:
        shutil.copy2(_REF_SRC, ref_dst)
        print(f"copied ref → {ref_dst.relative_to(_ROOT)}")
    elif dry:
        print(f"would copy ref → {ref_dst.relative_to(_ROOT)}")

    print(
        "\nLaunch:\n"
        f"  python scripts/run_paper_eval.py --suite {SUITE} "
        f"--gpus 0 --wait-deps --skip-rq-gate --lean\n"
        "  # or shard seeds across GPUs with --only …_seed44 …_seed45 …\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    materialize(dry=args.dry_run)


if __name__ == "__main__":
    main()
