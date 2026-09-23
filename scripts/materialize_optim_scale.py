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
* Probe suite ``optim_scale_probe``: skills-only seed=1 arms that keep the
  YAML outer (32/1024/50M/grad10) but ablate LR/ε — run these before/alongside
  the full tandem-first suite to see what breaks.

    python scripts/materialize_optim_scale.py
    python scripts/materialize_optim_scale.py --dry-run

    CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \\
      --suite optim_scale_probe --wait-deps --lean --skip-rq-gate
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
PROBE_SUITE = "optim_scale_probe"
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
SKILL_LR_SMOKE = 0.0005  # old healthy constant LR
MAX_GRAD_NORM = 10.0
EPS_START = 1.0
EPS_FINISH = 0.001
EPS_DECAY = 0.1
# Skills lock ε B (eps_sanity).
EPS_B_START = 0.2
EPS_B_FINISH = 0.01
EPS_B_DECAY = 0.2
OUT_MODELS = "./outputs/models/paper_eval/optim_scale"

SEEDS = (44, 45, 46)
PROBE_SEED = 1

MOD_KEYS = f10.MOD_KEYS

# Skills-only seed-1 probes: same outer as 1:1, ablate LR / ε.
# Order: control (full YAML) → isolate ε → isolate ε+old LR.
PROBE_ARMS: tuple[tuple[str, dict], ...] = (
    (
        "1to1",
        {
            "skill_lr": LR,
            "lr_linear_decay": True,
            "eps_start": EPS_START,
            "eps_finish": EPS_FINISH,
            "eps_decay": EPS_DECAY,
            "tag_lr": f"lr:{LR}",
            "tag_lr_decay": "lr_decay:linear",
            "tag_eps": "eps:1to0.001",
        },
    ),
    (
        "epsB",
        {
            "skill_lr": LR,
            "lr_linear_decay": True,
            "eps_start": EPS_B_START,
            "eps_finish": EPS_B_FINISH,
            "eps_decay": EPS_B_DECAY,
            "tag_lr": f"lr:{LR}",
            "tag_lr_decay": "lr_decay:linear",
            "tag_eps": "eps:B",
        },
    ),
    (
        "lr5e4_epsB",
        {
            "skill_lr": SKILL_LR_SMOKE,
            "lr_linear_decay": False,
            "eps_start": EPS_B_START,
            "eps_finish": EPS_B_FINISH,
            "eps_decay": EPS_B_DECAY,
            "tag_lr": f"lr:{SKILL_LR_SMOKE}",
            "tag_lr_decay": "lr_decay:none",
            "tag_eps": "eps:B",
        },
    ),
)


def skills_stem(seed: int) -> str:
    return f"optim_scale_skills_ego_seed{seed}"


def tandem_stem(seed: int) -> str:
    return f"optim_scale_tandem_pixel_ego_seed{seed}"


def probe_skills_stem(arm: str) -> str:
    return f"optim_scale_probe_skills_{arm}_seed{PROBE_SEED}"


def optim_scale_sq_stems() -> tuple[str, ...]:
    """Tandem trains first (OOM-fast), then skills, then home evals."""
    tandem_trains = tuple(tandem_stem(s) for s in SEEDS)
    skills_trains = tuple(skills_stem(s) for s in SEEDS)
    tandem_evals = tuple(f"{s}_eval" for s in tandem_trains)
    skills_evals = tuple(f"{s}_eval" for s in skills_trains)
    return tandem_trains + skills_trains + tandem_evals + skills_evals


def optim_scale_probe_stems() -> tuple[str, ...]:
    """Skills-only seed-1 HP probes (trains then home evals; no tandem)."""
    trains = tuple(probe_skills_stem(arm) for arm, _ in PROBE_ARMS)
    evals = tuple(f"{s}_eval" for s in trains)
    return trains + evals


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


def _patch_1to1(
    body: str,
    *,
    kind: str,
    suite: str = SUITE,
    skill_lr: float = LR,
    lr_linear_decay: bool = True,
    eps_start: float = EPS_START,
    eps_finish: float = EPS_FINISH,
    eps_decay: float = EPS_DECAY,
    tag_lr: str | None = None,
    tag_lr_decay: str | None = None,
    tag_eps: str | None = None,
    scale_tag: str = "pqn_optim_1to1",
) -> str:
    """Retarget final_10m body → YAML outer + chosen LR/ε/grad clip."""
    if tag_lr is None:
        tag_lr = f"lr:{skill_lr}"
    if tag_lr_decay is None:
        tag_lr_decay = "lr_decay:linear" if lr_linear_decay else "lr_decay:none"
    if tag_eps is None:
        tag_eps = f"eps:{eps_start}to{eps_finish}"

    reps = [
        ("  TOTAL_TIMESTEPS: 9994240", f"  TOTAL_TIMESTEPS: {TIMESTEPS}"),
        ("  NUM_ENVS: 128", f"  NUM_ENVS: {NUM_ENVS}"),
        ("  NUM_STEPS: 128", f"  NUM_STEPS: {NUM_STEPS}"),
        ("  NUM_EPOCHS: 4", f"  NUM_EPOCHS: {NUM_EPOCHS}"),
        ("  NUM_MINIBATCHES: 64", f"  NUM_MINIBATCHES: {NUM_MINIBATCHES}"),
        ("- budget:10m", f"- budget:{BUDGET_TAG}"),
        ("- suite:final_10m", f"- suite:{suite}"),
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

    # Shared outer HPs.
    body = _sub_alg_key(body, "NUM_STEPS", str(NUM_STEPS))
    body = _sub_alg_key(body, "NUM_ENVS", str(NUM_ENVS))
    body = _sub_alg_key(body, "NUM_EPOCHS", str(NUM_EPOCHS))
    body = _sub_alg_key(body, "NUM_MINIBATCHES", str(NUM_MINIBATCHES))
    body = _sub_alg_key(body, "TOTAL_TIMESTEPS", str(TIMESTEPS))
    body = _ensure_alg_key_after(
        body, "TOTAL_TIMESTEPS", "TOTAL_TIMESTEPS_DECAY", str(TIMESTEPS_DECAY)
    )
    body = _sub_alg_key(body, "LR", str(skill_lr))
    body = _sub_alg_key(body, "SKILL_LR", str(skill_lr))
    body = _sub_alg_key(body, "MAX_GRAD_NORM", str(MAX_GRAD_NORM))
    body = _ensure_alg_key_after(
        body,
        "MAX_GRAD_NORM",
        "LR_LINEAR_DECAY",
        "true" if lr_linear_decay else "false",
    )

    body = _sub_alg_key(body, "EPS_START", str(eps_start))
    body = _sub_alg_key(body, "EPS_FINISH", str(eps_finish))
    body = _sub_alg_key(body, "EPS_DECAY", str(eps_decay))

    if kind == "tandem":
        body = _sub_alg_key(body, "META_LR", str(skill_lr))
        body = _sub_alg_key(body, "SKILL_EPS_START", str(eps_start))
        body = _sub_alg_key(body, "SKILL_EPS_FINISH", str(eps_finish))
        body = _sub_alg_key(body, "SKILL_EPS_DECAY", str(eps_decay))
        body = _sub_alg_key(body, "META_EPS_START", str(eps_start))
        body = _sub_alg_key(body, "META_EPS_FINISH", str(eps_finish))
        body = _sub_alg_key(body, "META_EPS_DECAY", str(eps_decay))

    for drop in (
        "- lock:skills_lr1e3\n",
        "- lock:skills_lr5e4\n",
        "- lock:tandem_e01\n",
        "- lr:constant\n",
    ):
        body = body.replace(drop, "")
    # Drop any prior scale tag block so re-materialize is idempotent.
    body = re.sub(
        r"(?m)^- scale:pqn_optim(?:_1to1)?\n"
        r"(?:- num_envs:\d+\n)?"
        r"(?:- num_steps:\d+\n)?"
        r"(?:- num_updates:\d+\n)?"
        r"(?:- num_epochs:\d+\n)?"
        r"(?:- num_minibatches:\d+\n)?"
        r"(?:- lr:[^\n]+\n)?"
        r"(?:- lr_decay:[^\n]+\n)?"
        r"(?:- max_grad_norm:[^\n]+\n)?"
        r"(?:- eps:[^\n]+\n)?"
        r"(?:- probe:[^\n]+\n)?",
        "",
        body,
    )
    needle = f"- budget:{BUDGET_TAG}\n"
    insert = (
        f"- budget:{BUDGET_TAG}\n"
        f"- scale:{scale_tag}\n"
        f"- num_envs:{NUM_ENVS}\n"
        f"- num_steps:{NUM_STEPS}\n"
        f"- num_updates:{NUM_UPDATES}\n"
        f"- num_epochs:{NUM_EPOCHS}\n"
        f"- num_minibatches:{NUM_MINIBATCHES}\n"
        f"- {tag_lr}\n"
        f"- {tag_lr_decay}\n"
        f"- max_grad_norm:{int(MAX_GRAD_NORM)}\n"
        f"- {tag_eps}\n"
    )
    if needle in body and f"- scale:{scale_tag}\n" not in body:
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
    suite: str = SUITE,
    patch_kwargs: dict | None = None,
) -> str:
    body = _patch_1to1(body, kind=kind, suite=suite, **(patch_kwargs or {}))
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


def _probe_skills_train(arm: str, hp: dict) -> str:
    stem = probe_skills_stem(arm)
    base = f10._skills_train("seaquest")
    lr = hp["skill_lr"]
    decay = "linear" if hp["lr_linear_decay"] else "const"
    return _retarget(
        base,
        stem=stem,
        seed=PROBE_SEED,
        suite=PROBE_SUITE,
        patch_kwargs={
            "skill_lr": hp["skill_lr"],
            "lr_linear_decay": hp["lr_linear_decay"],
            "eps_start": hp["eps_start"],
            "eps_finish": hp["eps_finish"],
            "eps_decay": hp["eps_decay"],
            "tag_lr": hp["tag_lr"],
            "tag_lr_decay": hp["tag_lr_decay"],
            "tag_eps": hp["tag_eps"],
            "scale_tag": f"pqn_optim_probe_{arm}",
        },
        header=(
            f"# Seaquest ego skills PROBE seed={PROBE_SEED} arm={arm}: "
            f"outer 1:1 ({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}, grad={MAX_GRAD_NORM:g}); "
            f"SKILL_LR={lr} ({decay}); "
            f"ε {hp['eps_start']}→{hp['eps_finish']}/decay={hp['eps_decay']}."
        ),
        run_name=f"sq_{PROBE_SUITE}_skills_{arm}_seed{PROBE_SEED}_{BUDGET_TAG}",
        kind="skills",
    )


def _probe_skills_eval(arm: str, hp: dict) -> str:
    stem = probe_skills_stem(arm)
    base = f10._skills_eval("seaquest")
    return _retarget(
        base,
        stem=stem,
        seed=PROBE_SEED,
        suite=PROBE_SUITE,
        patch_kwargs={
            "skill_lr": hp["skill_lr"],
            "lr_linear_decay": hp["lr_linear_decay"],
            "eps_start": hp["eps_start"],
            "eps_finish": hp["eps_finish"],
            "eps_decay": hp["eps_decay"],
            "tag_lr": hp["tag_lr"],
            "tag_lr_decay": hp["tag_lr_decay"],
            "tag_eps": hp["tag_eps"],
            "scale_tag": f"pqn_optim_probe_{arm}",
        },
        header=f"# Home mod eval — optim_scale probe skills {arm} seed{PROBE_SEED}.",
        run_name=f"sq_{PROBE_SUITE}_eval_skills_{arm}_seed{PROBE_SEED}_{BUDGET_TAG}",
        kind="skills",
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
            "note: suite order is tandem-first (OOM-fast); use optim_scale_probe for skills-first HP checks",
            f"outputs_root: {OUT_MODELS}",
            "stems:",
            *[f"  - {s}" for s in optim_scale_sq_stems()],
        ]
    )


def _probe_manifest() -> str:
    lines = [
        "# optim_scale_probe inventory (not runnable).",
        f"suite: {PROBE_SUITE}",
        f"budget_tag: {BUDGET_TAG}",
        f"seed: {PROBE_SEED}",
        "kind: skills_only",
        f"outer: {NUM_ENVS}x{NUM_STEPS}x{NUM_UPDATES} T={TIMESTEPS} max_grad={MAX_GRAD_NORM}",
        "arms:",
    ]
    for arm, hp in PROBE_ARMS:
        lines.append(
            f"  - {arm}: lr={hp['skill_lr']} decay={hp['lr_linear_decay']} "
            f"eps={hp['eps_start']}->{hp['eps_finish']}/{hp['eps_decay']}"
        )
    lines.append("stems:")
    lines.extend(f"  - {s}" for s in optim_scale_probe_stems())
    return f10._lines(lines)


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

    print(
        f"optim_scale_probe → skills-only seed={PROBE_SEED} arms="
        f"{[a for a, _ in PROBE_ARMS]}"
    )
    for arm, hp in PROBE_ARMS:
        stem = probe_skills_stem(arm)
        pairs.append((_OUT / f"{stem}.yaml", _probe_skills_train(arm, hp)))
        pairs.append((_OUT / f"{stem}_eval.yaml", _probe_skills_eval(arm, hp)))
    pairs.append((_OUT / "probe_manifest.yaml", _probe_manifest()))

    for path, body in pairs:
        f10._write(path, body, dry=dry)

    ref_dst = _OUT / "pqn_jaxatari_pixel_optimal_scaling.yaml"
    if _REF_SRC.is_file() and not dry:
        shutil.copy2(_REF_SRC, ref_dst)
        print(f"copied ref → {ref_dst.relative_to(_ROOT)}")
    elif dry:
        print(f"would copy ref → {ref_dst.relative_to(_ROOT)}")

    print(
        "\nLaunch probes first (skills-only):\n"
        f"  python scripts/run_paper_eval.py --suite {PROBE_SUITE} "
        f"--gpus 0 --wait-deps --skip-rq-gate --lean\n"
        "Full 1:1 suite (tandem-first):\n"
        f"  python scripts/run_paper_eval.py --suite {SUITE} "
        f"--gpus 0 --wait-deps --skip-rq-gate --lean\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    materialize(dry=args.dry_run)


if __name__ == "__main__":
    main()
