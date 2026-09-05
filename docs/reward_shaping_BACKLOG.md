# Reward shaping backlog

Future experiment ideas and known obs–reward mismatches.

## COLLECT / NAVIGATE / AVOID approach distance — centre vs edge-to-edge

**Status:** done in code (2026-09-05) — retrain required

**Was:** Dense COLLECT / NAVIGATE / AVOID shaping used centre-to-centre Euclidean
distance (`_centre_distances`). Ego observations use AABB surface gap
(`aabb_surface_gap_px`).

**Now:** `_approach_distances` in `skill_rewards.py` uses the same
`aabb_surface_gap_px` as ego (including Freeway bottom-edge y). DESTROY standoff
melee range also uses surface gap; DESTROY y-align still uses centre `|Δy|`.

**Retrain suites:**

```bash
# 2 GPUs — Kangaroo (clip50+θ8) + Seaquest navigatesurface_surfacegap → eval → Pong
python scripts/run_paper_eval.py --suite surfacegap_dual --gpus 0,1 --wait-deps --lean

# Pieces alone
python scripts/run_paper_eval.py --suite kangaroo_surfacegap --gpus 0 --wait-deps --lean
python scripts/run_paper_eval.py --suite navigate_surface_surfacegap --gpus 0 --wait-deps --lean
```
