# Collect sparse Arms 1–2 — experiment log

Local notes for the sequential reward ablation started after honest-transfer
Pong diagnosis (facing φ8 clip50 v2). Not paper prose — just so later eval
still makes sense.

## Problem

SeaQuest skills look strong / robust, but SeaQuest→Pong under the **honest**
facing-relative mapper fails to intercept cleanly.

Diagnosis on `seaquest_skills_ego_pov_clip50_facing_psibins8_v2`
(`tmp/pong_transfer_diag/v2_phi8.json`):

- ~85% forwardish ego actions; ~42% map to NOOP on Pong (FORWARD into field).
- ~75% idle while laterally misaligned (`|lat|≥8`).
- Mean `|lat|` near the paddle ≈ **16px** (matches “starts late, stops ~10px short”).
- `EGO_LEGAL_ACTION_MASK` at eval kills FORWARD→NOOP (~67% argmax flips) but
  does **not** fix the miss distance — so mask alone is not enough.

Hypothesis: dense COLLECT approach shaping (`+0.01` when centre-dist shrinks)
encourages local greedy / late-threshold behaviour that does not transfer to
fixed-facing lateral intercept.

## What we are testing (Arms 1–2 only for this run)

Spine: same as `seaquest_skills_ego_pov_clip50_facing_psibins8_v2`
(POV mapper + facing frame + φ8 + clip50). **No ego encoding change.**

| Arm | Stem suffix | COLLECT | DESTROY | Death on COLLECT |
|-----|-------------|---------|---------|------------------|
| **1** | `…_v2_collect_sparse` | sparse only (`COLLECT_SPARSE_ONLY`) | living tax + contact-fail **on** | **on** |
| **2** | `…_v2_collect_sparse_destroy_sparse` | sparse only | living tax + contact-fail **off** (kill +1 only) | **on** |

Suite (sequential, wait-deps): `collect_sparse_arms`

Per arm: SeaQuest train (~25M) → SeaQuest full mod eval @27k → Pong base zero-shot.

## Why this order / design

1. Arm1 removes *bias toward closing*, without inventing Pong-specific shaping
   (no lateral-align reward). Keeps DESTROY urgency so RQ1 isn’t a full rewrite.
2. Arm2 asks whether DESTROY dense terms were carrying source-task performance
   or interacting badly with COLLECT credit.
3. Death penalty on COLLECT kept for both (honest miss signal on Pong). A later
   Arm 2b (`SKILL_DEATH_PENALIZE_COLLECT: false`) is **not** in this suite —
   only if 1/2 show Pong signal.

Explicitly **not** in this run: legal-mask train, L/C/R×near/far ego, 4-way φ,
COLLECT living tax, lateral-align shaping.

## How to run (cluster)

```bash
python scripts/materialize_collect_sparse_arms.py   # idempotent
bash scripts/run_collect_sparse_arms.sh --gpus 0 --wait-deps --lean
# equivalent:
# python scripts/run_paper_eval.py --suite collect_sparse_arms --gpus 0 --wait-deps --lean
```

One GPU is enough: arms are sequential (Arm2 does not need Arm1 ckpt, but we
serialize to keep load/logs simple while away).

## Success / gate criteria (for later)

- SeaQuest: not catastrophic vs v2 baseline (~19k-ish base; Arm2 may drop).
- Pong: mean ≫ −21 and/or player score > 0 on base; compare to v2 (~+1 / 21–20).
- If Arm1 helps Pong → proceed to legal-mask train / restrained ego.
- If both fail Pong but SeaQuest holds → reward sparsity alone wasn’t the lever.

## Code touchpoints

- Flag: `COLLECT_SPARSE_ONLY` in `skill_rewards.py` (wins over living tax).
- Configs: `scripts/materialize_collect_sparse_arms.py`
- Suite: `COLLECT_SPARSE_ARMS_SUITE` in `scripts/run_paper_eval.py`
- Diagnosis: `scripts/diagnose_pong_transfer.py`, `tmp/pong_transfer_diag/`

## Planned later (not wired)

3. Restrained ego L/C/R × near/far (all skills)
4. 4-way facing angle (± near/far)
5. Optional: drop COLLECT death if 1/2 work
6. Optional parallel: train with `EGO_LEGAL_ACTION_MASK: true` on current spine
