#!/usr/bin/env bash
# Post-facingfix Seaquest skill retrain + Pong transfer eval.
#
# Prerequisites:
#   1. Facing-frame fix merged (obs/mapper share view-axis table).
#   2. Configs materialized:  python scripts/materialize_pov_facing_v2_configs.py --phase 1
#
# Phase 1 (recommended first — ~2 GPU-nights for the two Seaquest trains):
#   - seaquest_skills_ego_pov_clip50_facing_psibins8_v2      (φ-bins8 baseline)
#   - seaquest_skills_ego_pov_clip50_facing_psibins8_mid_v2  (+ soft midfield COLLECT)
#   Then Seaquest mod eval @27k + Pong zero-shot for both.
#
# Phase 2 (after Pong base return looks sane in visualizer):
#   - facing_xy pose encoding + Beamrider transfer sanity checks
#
# Usage:
#   # Materialize YAMLs (safe to re-run):
#   python scripts/materialize_pov_facing_v2_configs.py --phase 1
#
#   # Train only (2 GPUs, parallel):
#   bash scripts/run_pov_facing_v2_retrain.sh train --gpus 0,1 --lean
#
#   # Full phase-1 pipeline (train → SQ eval → Pong transfer), workers wait for ckpts:
#   bash scripts/run_pov_facing_v2_retrain.sh phase1 --gpus 0,1 --wait-deps --lean
#
#   # Phase 2 (after materialize --phase all):
#   python scripts/materialize_pov_facing_v2_configs.py --phase all
#   bash scripts/run_pov_facing_v2_retrain.sh phase2 --gpus 0,1 --wait-deps --lean
#
# See docs/POV_FRAME_BACKLOG.md for open move-table fixes (tennis, skiing, etc.).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER=("$PYTHON_BIN" scripts/run_paper_eval.py)

phase="${1:-phase1}"
shift || true

case "$phase" in
  train)
    exec "${RUNNER[@]}" --suite pov_facing_v2_train "$@"
    ;;
  phase1)
    exec "${RUNNER[@]}" --suite pov_facing_v2 --wait-deps "$@"
    ;;
  phase2)
    exec "${RUNNER[@]}" --suite pov_facing_v2_phase2 --wait-deps "$@"
    ;;
  materialize)
  mat)
    exec "$PYTHON_BIN" scripts/materialize_pov_facing_v2_configs.py "${1:-all}"
    ;;
  *)
    echo "usage: $0 {train|phase1|phase2|materialize} [run_paper_eval.py args...]" >&2
    exit 2
    ;;
esac
