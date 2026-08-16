#!/usr/bin/env bash
# Skill-reshape validation suite after COLLECT approach + AVOID escape shaping.
#
# 1) Train RQ1 Seaquest skills/ego
# 2) Eval Seaquest mods (standard ego eval)
# 3) AVOID probe: same ckpt, HEURISTIC_MODE=avoid_only (threat-forced targeting)
# 4) Zero-shot Pong transfer (COLLECT approach shaping check)
# 5) Zero-shot BeamRider transfer (NO_TARGET_GAME_ACTION=FIRE for level advance)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_skill_reshape_suite.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_skill_reshape_suite.sh --trial   # short smoke
#   bash scripts/run_skill_reshape_suite.sh --eval-only   # skip train (ckpt must exist)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:src"
JAX="${PAPER_EVAL_JAXATARI_ROOT:-/home/paul/Documents/JAXAtari-seaquest-robustness}"
EXTRA=()
EVAL_ONLY=0
for a in "$@"; do
  if [[ "$a" == "--eval-only" ]]; then
    EVAL_ONLY=1
  else
    EXTRA+=("$a")
  fi
done

COMMON=(
  --jaxatari-root "$JAX"
  --wandb-mode "${WANDB_MODE:-online}"
  --eval-max-frames "${EVAL_MAX_FRAMES:-27000}"
  "${EXTRA[@]}"
)

if [[ "$EVAL_ONLY" -eq 0 ]]; then
  echo "=== 1/5 train seaquest_skills_ego ==="
  python scripts/run_paper_eval.py --phase train --only seaquest_skills_ego "${COMMON[@]}"
else
  echo "=== 1/5 skip train (--eval-only) ==="
fi

echo "=== 2/5 eval Seaquest ego mods ==="
python scripts/run_paper_eval.py --phase eval --only seaquest_skills_ego_eval "${COMMON[@]}"

echo "=== 3/5 AVOID probe (heuristic avoid_only) ==="
python scripts/run_paper_eval.py --phase eval --only seaquest_skills_ego_avoid_probe "${COMMON[@]}"

echo "=== 4/5 Pong zero-shot transfer ==="
python scripts/run_paper_eval.py --phase eval --only pong_transfer_ego "${COMMON[@]}"

echo "=== 5/5 BeamRider zero-shot transfer (idle FIRE) ==="
python scripts/run_paper_eval.py --phase eval --only beamrider_transfer_ego "${COMMON[@]}"

echo
echo "Done. Compare:"
echo "  - Seaquest base/mod scores vs prior ego ckpt"
echo "  - avoid_probe JSON: higher steps / fewer life losses under forced AVOID targets"
echo "  - pong_transfer_ego base mean (was ~-21)"
echo "  - beamrider_transfer_ego base mean + videos past level-1 gate"
echo "Artifacts: outputs/models/paper_eval/eval/*_mods.json"
