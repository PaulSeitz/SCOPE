# Restrained ego pose (Arm 3) — experiment log

## Motivation

SeaQuest→Pong transfer with continuous φ8 + clip50 overfits SeaQuest situations.
Arm 3 replaces continuous distance/angle with a **finite alphabet**:

- **Lateral:** left / center / right (|lat| < 8px → center)
- **Distance:** near / far (dist ≤ 50px → near)

Pose = 5 one-hot channels per frame (3 lateral + 2 distance). Same v2 reward
spine as the φ8 baseline (approach-shaped COLLECT; sparse arms rejected).

## Legal ego mask

`EGO_LEGAL_ACTION_MASK: true` on train + Pong transfer. Masks Q-values for ego
actions that map to NOOP under the honest facing mapper (e.g. FORWARD on Pong).
Defensible for zero-shot transfer validity.

## Suite

Materialize + run:

```bash
python scripts/materialize_restrained_ego_suite.py
bash scripts/run_restrained_ego_suite.sh --gpus 0 --wait-deps --lean
```

Stems (`restrained_ego` suite):

1. `seaquest_skills_ego_pov_facing_lcr_nearfar_v2` — train
2. `seaquest_skills_ego_pov_facing_lcr_nearfar_v2_eval_full` — Seaquest mods
3. `pong_transfer_ego_pov_facing_lcr_nearfar_v2` — Pong zero-shot

## Gates

| Metric | φ8 v2 baseline | Arm3 target |
|--------|----------------|-------------|
| SeaQuest base | ~22.6k | ≥20k (don't collapse source) |
| Pong base | +1 (mask off) | > baseline with mask on |

## Results (cluster1, synced 2026-09-05)

**Fail — do not iterate on this encoding.**

| Metric | φ8 v2 | Arm3 LCR×near/far |
|--------|-------|-------------------|
| SeaQuest base | **22 600** | **2 643** (−88%) |
| SeaQuest median retention | 0.936 | 0.953 (on a collapsed base) |
| Pong base | **+1** (21–20) | **−21** (0–21 shutout) |

Source competence collapsed: dynamics/semantic mods like `faster_enemies`,
`lethal_divers`, `gravity` fall to ~140–230. Farming mods still inflate
retention ratios — ignore those.

Coarse pose + legal mask **did not** unlock Pong intercept; it destroyed the
SeaQuest skill set first. Next transfer work should keep continuous φ8 (or
navigatesurface) and attack action/legal-mask / reward–obs consistency instead.

See `docs/transfer_ablation_ANALYSIS.md` for the full ablation board.
