#!/usr/bin/env python3
"""Materialize Seaquest ego-skills + tandem-pixel at PQN optimal-scaling 1:1 HPs.

Reference (flat PQN pixel)::

    /home/paul/Downloads/pqn_jaxatari_pixel_optimal_scaling.yaml
    → TOTAL_TIMESTEPS=5e7, TOTAL_TIMESTEPS_DECAY=5e7,
      NUM_ENVS=1024, NUM_STEPS=32, NUM_EPOCHS=2, NUM_MINIBATCHES=256,
      EPS 1→0.001 over EPS_DECAY=0.1 of decay updates,
      LR=2.5e-4 + LR_LINEAR_DECAY, MAX_GRAD_NORM=10, GAMMA=0.99, LAMBDA=0.65.

CBL mapping (true 1:1 on shared HPs)
------------------------------------
* Outer: 1024 envs × NUM_STEPS=32 → NUM_UPDATES = floor(5e7/(1024×32)) = 1525
  → TOTAL_TIMESTEPS kept at 50_000_000 (same as YAML; trainers floor to 1525 updates).
* Skills: SKILL_LR=2.5e-4 + LR_LINEAR_DECAY (→1e-20 over TOTAL_TIMESTEPS_DECAY).
* Tandem: same for SKILL_LR and META_LR; same ε schedule on skill + meta heads.
* MAX_GRAD_NORM=10 (YAML), not the hierarchical template default 0.8.
* 3 train seeds via separate stems (44/45/46).

    python scripts/materialize_optim_scale.py
    python scripts/materialize_optim_scale.py --dry-run

    CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \\
      --suite optim_scale_sq --wait-deps --lean --skip-rq-gate
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import materialize_final_10m as f10

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "configs/experiment/paper_eval/optim_scale"
_REF_SRC = Path("/home/paul/Downloads/pqn_jaxatari_pixel_optimal_scaling.yaml")

SUITE = "optim_scale_sq"
BUDGET_TAG = "50m_optim"
# Match YAML literally; NUM_UPDATES = TOTAL // (NUM_ENVS * NUM_STEPS).
TIMESTEPS = 50_000_000
TIMESTEPS_DECAY = 50_000_000
NUM_ENVS = 1024
NUM_STEPS = 32
NUM_UPDATES = TIMESTEPS // (NUM_ENVS * NUM_STEPS)  # 1525
NUM_EPOCHS = 2
NUM_MINIBATCHES = 256
LR = 0.00025
MAX_GRAD_NORM = 10.0
EPS_START = 1.0
EPS_FINISH = 0.001
EPS_DECAY = 0.1
OUT_MODELS = "./outputs/models/paper_eval/optim_scale"

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


def _sub_alg_key(body: str, key: str, value: str) -> str:
    """Replace ``  KEY: ...`` under alg (2-space indent) if present; else no-op."""
    pat = re.compile(rf"(?m)^  {re.escape(key)}:.*$")
    repl = f"  {key}: {value}"
    if pat.search(body):
        return pat.sub(repl, body, count=1)
    return body


def _ensure_alg_key_after(body: str, after_key: str, key: str, value: str) -> str:
    """Insert ``  KEY: value`` after ``after_key`` if ``key`` is missing."""
    if re.search(rf"(?m)^  {re.escape(key)}:", body):
        return _sub_alg_key(body, key, value)
    pat = re.compile(rf"(?m)^(  {re.escape(after_key)}:.*)$")
    m = pat.search(body)
    if not m:
        return body
    insert = f"{m.group(1)}\n  {key}: {value}"
    return pat.sub(insert, body, count=1)


def _patch_1to1(body: str, *, kind: str) -> str:
    """Retarget final_10m body → YAML-matched outer + LR/ε/grad clip."""
    reps = [
        ("  TOTAL_TIMESTEPS: 9994240", f"  TOTAL_TIMESTEPS: {TIMESTEPS}"),
        ("  NUM_ENVS: 128", f"  NUM_ENVS: {NUM_ENVS}"),
        ("  NUM_STEPS: 128", f"  NUM_STEPS: {NUM_STEPS}"),
        ("  NUM_EPOCHS: 4", f"  NUM_EPOCHS: {NUM_EPOCHS}"),
        ("  NUM_MINIBATCHES: 64", f"  NUM_MINIBATCHES: {NUM_MINIBATCHES}"),
        ("- budget:10m", f"- budget:{BUDGET_TAG}"),
        ("- suite:final_10m", f"- suite:{SUITE}"),
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

    # Shared 1:1 HPs (skills + tandem).
    body = _sub_alg_key(body, "NUM_STEPS", str(NUM_STEPS))
    body = _sub_alg_key(body, "NUM_ENVS", str(NUM_ENVS))
    body = _sub_alg_key(body, "NUM_EPOCHS", str(NUM_EPOCHS))
    body = _sub_alg_key(body, "NUM_MINIBATCHES", str(NUM_MINIBATCHES))
    body = _sub_alg_key(body, "TOTAL_TIMESTEPS", str(TIMESTEPS))
    body = _ensure_alg_key_after(
        body, "TOTAL_TIMESTEPS", "TOTAL_TIMESTEPS_DECAY", str(TIMESTEPS_DECAY)
    )
    body = _sub_alg_key(body, "LR", str(LR))
    body = _sub_alg_key(body, "SKILL_LR", str(LR))
    body = _sub_alg_key(body, "MAX_GRAD_NORM", str(MAX_GRAD_NORM))
    body = _ensure_alg_key_after(body, "MAX_GRAD_NORM", "LR_LINEAR_DECAY", "true")

    # ε schedule = YAML (skills head; tandem also sets skill/meta ε below).
    body = _sub_alg_key(body, "EPS_START", str(EPS_START))
    body = _sub_alg_key(body, "EPS_FINISH", str(EPS_FINISH))
    body = _sub_alg_key(body, "EPS_DECAY", str(EPS_DECAY))

    if kind == "tandem":
        body = _sub_alg_key(body, "META_LR", str(LR))
        body = _sub_alg_key(body, "SKILL_EPS_START", str(EPS_START))
        body = _sub_alg_key(body, "SKILL_EPS_FINISH", str(EPS_FINISH))
        body = _sub_alg_key(body, "SKILL_EPS_DECAY", str(EPS_DECAY))
        body = _sub_alg_key(body, "META_EPS_START", str(EPS_START))
        body = _sub_alg_key(body, "META_EPS_FINISH", str(EPS_FINISH))
        body = _sub_alg_key(body, "META_EPS_DECAY", str(EPS_DECAY))

    # Tags: drop old lock / constant-lr; stamp 1:1.
    for drop in (
        "- lock:skills_lr1e3\n",
        "- lock:skills_lr5e4\n",
        "- lock:tandem_e01\n",
        "- lr:constant\n",
    ):
        body = body.replace(drop, "")
    needle = f"- budget:{BUDGET_TAG}\n"
    insert = (
        f"- budget:{BUDGET_TAG}\n"
        f"- scale:pqn_optim_1to1\n"
        f"- num_envs:{NUM_ENVS}\n"
        f"- num_steps:{NUM_STEPS}\n"
        f"- num_updates:{NUM_UPDATES}\n"
        f"- num_epochs:{NUM_EPOCHS}\n"
        f"- num_minibatches:{NUM_MINIBATCHES}\n"
        f"- lr:{LR}\n"
        f"- lr_decay:linear\n"
        f"- max_grad_norm:{int(MAX_GRAD_NORM)}\n"
        f"- eps:1to0.001\n"
    )
    if needle in body and "- scale:pqn_optim_1to1\n" not in body:
        # Remove any prior scale:pqn_optim block tags from older materializations.
        body = re.sub(
            r"(?m)^- scale:pqn_optim\n"
            r"(?:- num_envs:\d+\n)?"
            r"(?:- num_steps:\d+\n)?"
            r"(?:- num_updates:\d+\n)?"
            r"(?:- num_epochs:\d+\n)?"
            r"(?:- num_minibatches:\d+\n)?",
            "",
            body,
        )
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
    body = _patch_1to1(body, kind=kind)
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
    body = body.replace(f"- {f10.STEM_SKILLS}\n", f"- {stem}\n")
    body = body.replace(f"- {f10.STEM_TANDEM_PIXEL}\n", f"- {stem}\n")
    body = body.replace(f"{f10.OUT_MODELS}/", f"{OUT_MODELS}/")
    body = body.replace("PLACEHOLDER", f"{OUT_MODELS}/{stem}")
    return body


def _skills_train(seed: int) -> str:
    stem = skills_stem(seed)
    base = f10._skills_train("seaquest")
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        header=(
            f"# Seaquest ego skills — PQN optimal-scaling 1:1 "
            f"({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}; LR={LR}+linear decay; "
            f"MAX_GRAD_NORM={MAX_GRAD_NORM:g}; ε {EPS_START}→{EPS_FINISH}); "
            f"seed={seed}."
        ),
        run_name=f"sq_{SUITE}_skills_ego_seed{seed}_{BUDGET_TAG}_1to1",
        kind="skills",
    )


def _skills_eval(seed: int) -> str:
    stem = skills_stem(seed)
    base = f10._skills_eval("seaquest")
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        header=f"# Home mod eval — optim_scale 1:1 skills seed{seed}.",
        run_name=f"sq_{SUITE}_eval_skills_ego_seed{seed}_{BUDGET_TAG}_1to1",
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
            f"# Seaquest tandem pixel — PQN optimal-scaling 1:1 "
            f"({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}; LR={LR}+linear decay "
            f"skill+meta; MAX_GRAD_NORM={MAX_GRAD_NORM:g}); seed={seed}. "
            f"Soft risk: pixel meta @ {NUM_ENVS} envs may OOM."
        ),
        run_name=f"sq_{SUITE}_tandem_pixel_seed{seed}_{BUDGET_TAG}_1to1",
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
        header=f"# Home mod eval — optim_scale 1:1 tandem pixel seed{seed}.",
        run_name=f"sq_{SUITE}_eval_tandem_pixel_seed{seed}_{BUDGET_TAG}_1to1",
        kind="tandem",
    )


def _manifest() -> str:
    return f10._lines(
        [
            "# optim_scale_sq inventory (not runnable).",
            f"suite: {SUITE}",
            f"budget_tag: {BUDGET_TAG}",
            f"total_timesteps: {TIMESTEPS}",
            f"total_timesteps_decay: {TIMESTEPS_DECAY}",
            f"num_envs: {NUM_ENVS}",
            f"num_steps: {NUM_STEPS}",
            f"num_updates: {NUM_UPDATES}",
            f"num_epochs: {NUM_EPOCHS}",
            f"num_minibatches: {NUM_MINIBATCHES}",
            f"lr: {LR}",
            "lr_linear_decay: true",
            f"max_grad_norm: {MAX_GRAD_NORM}",
            f"eps: {EPS_START} -> {EPS_FINISH} (decay={EPS_DECAY})",
            f"seeds: {list(SEEDS)}",
            "ref: pqn_jaxatari_pixel_optimal_scaling.yaml (1:1 shared HPs)",
            "skills: SKILL_LR + LR_LINEAR_DECAY + YAML ε",
            "tandem: META_LR=SKILL_LR + same decay + YAML ε on skill+meta",
            f"outputs_root: {OUT_MODELS}",
            "stems:",
            *[f"  - {s}" for s in optim_scale_sq_stems()],
        ]
    )


def materialize(*, dry: bool) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"optim_scale_sq 1:1 → {_OUT.relative_to(_ROOT)}  "
        f"T={TIMESTEPS} envs={NUM_ENVS} steps={NUM_STEPS} updates={NUM_UPDATES} "
        f"LR={LR} max_grad={MAX_GRAD_NORM} seeds={SEEDS}"
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
