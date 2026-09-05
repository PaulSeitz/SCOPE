# Reward shaping backlog

Future experiment ideas and known obs–reward mismatches.

## COLLECT / NAVIGATE / AVOID approach distance — centre vs edge-to-edge

**Status:** done in code (2026-09-05) — retrain required

**Was:** Dense COLLECT / NAVIGATE / AVOID shaping used centre-to-centre Euclidean
distance (`_centre_distances`). Ego observations use AABB surface gap
(`aabb_surface_gap_px`).

**Now:** `_approach_distances` uses `aabb_geom_distance_px` (same `EGO_AABB_GEOM`
as ego). Default `centre_ray` = `aabb_surface_gap_px`. Optional `closest_points`
= true edge–edge separation. DESTROY y-align still uses centre `|Δy|` and stays off
on transfer spines.

**Pose ablations (obs–reward aligned, navigatesurface_surfacegap spine):**

```bash
# 2 GPUs — hit-corridor vs closest-points φ in parallel → eval → Pong
python scripts/run_paper_eval.py --suite surfacegap_geom_dual --gpus 0,1 --wait-deps --lean
```

- `USE_EGO_HIT_CORRIDOR`: +`lat_rel`, +`in_corridor` (body-relative lateral slab)
- `EGO_AABB_GEOM: closest_points`: distance + φ from true AABB closest-points vector
  (centre-ray surface points share centre bearing — φ only changes with closest-points)

**Older surfacegap suites:**

```bash
python scripts/run_paper_eval.py --suite surfacegap_dual --gpus 0,1 --wait-deps --lean
python scripts/run_paper_eval.py --suite kangaroo_surfacegap --gpus 0 --wait-deps --lean
python scripts/run_paper_eval.py --suite navigate_surface_surfacegap --gpus 0 --wait-deps --lean
```
