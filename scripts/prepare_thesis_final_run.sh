#!/usr/bin/env bash
# Archive (and optionally remove) artifacts for a clean thesis tier run.
#
# Usage:
#   bash scripts/prepare_thesis_final_run.sh --archive-only
#   bash scripts/prepare_thesis_final_run.sh              # archive + delete ckpts/evals
#
# Then Tier A on 2 GPUs:
#   python scripts/run_paper_eval.py --suite thesis_tier_a --gpus 0,1 --wait-deps --lean \\
#     --batch-tag thesis_tier_a_$(date +%Y%m%d)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ARCHIVE_ONLY=false
if [[ "${1:-}" == "--archive-only" ]]; then
  ARCHIVE_ONLY=true
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$ROOT/outputs/models/paper_eval/archive/thesis_${STAMP}"
EVAL="$ROOT/outputs/models/paper_eval/eval"

TIER_A_EVAL_STEMS=(
  seaquest_skills_ego_clip50_angbins_eval_full
  seaquest_skills_pixel_eval
  seaquest_skills_oc_eval
  seaquest_pqn_eval
  seaquest_pqn_oc_eval
  seaquest_heuristic_random
  pong_transfer_ego
  tennis_transfer_ego
  beamrider_transfer_ego
  freeway_transfer_ego
  asterix_transfer_ego
  phoenix_transfer_ego
  seaquest_tandem_pixel_ego_clip50_angbins_eval
)

TRAIN_DIRS=(
  outputs/models/paper_eval/rq1/seaquest_skills_ego_clip50_angbins
  outputs/models/paper_eval/rq1/seaquest_skills_pixel
  outputs/models/paper_eval/rq1/seaquest_skills_oc
  outputs/models/paper_eval/rq1/seaquest_pqn
  outputs/models/paper_eval/rq1/seaquest_pqn_oc
  outputs/models/paper_eval/rq3/seaquest_tandem_pixel_ego_clip50_angbins
)

mkdir -p "$ARCHIVE/eval" "$ARCHIVE/train"

echo "Archiving thesis artifacts → $ARCHIVE"

for stem in "${TIER_A_EVAL_STEMS[@]}"; do
  for f in "$EVAL/${stem}"*.json "$EVAL/${stem}"*.csv; do
    [[ -e "$f" ]] || continue
    cp -a "$f" "$ARCHIVE/eval/"
    echo "  archived eval/$(basename "$f")"
  done
done

for rel in "${TRAIN_DIRS[@]}"; do
  if [[ -d "$ROOT/$rel" ]]; then
    cp -a "$ROOT/$rel" "$ARCHIVE/train/"
    echo "  archived train/$(basename "$rel")"
  fi
done

if [[ -f "$EVAL/all_mod_eval_summary.csv" ]]; then
  cp -a "$EVAL/all_mod_eval_summary.csv" "$ARCHIVE/eval/"
  cp -a "$EVAL/all_mod_eval_episodes.csv" "$ARCHIVE/eval/" 2>/dev/null || true
fi

if $ARCHIVE_ONLY; then
  echo "Archive-only mode (nothing deleted)."
  exit 0
fi

echo ""
echo "Removing Tier A train ckpts and eval outputs..."
for rel in "${TRAIN_DIRS[@]}"; do
  [[ -d "$ROOT/$rel" ]] && rm -rf "$ROOT/$rel" && echo "  removed $rel"
done
for stem in "${TIER_A_EVAL_STEMS[@]}"; do
  rm -f "$EVAL/${stem}"*.json "$EVAL/${stem}"*.csv 2>/dev/null || true
done
rm -f "$EVAL/all_mod_eval_summary.csv" "$EVAL/all_mod_eval_episodes.csv" 2>/dev/null || true

echo ""
echo "Ready. Launch Tier A:"
echo "  python scripts/run_paper_eval.py --suite thesis_tier_a --gpus 0,1 --wait-deps --lean \\"
echo "    --batch-tag thesis_tier_a_${STAMP%%_*}"
