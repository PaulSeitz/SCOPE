#!/usr/bin/env bash
# Record RQ2 zero-shot transfer rollouts on base (no mods) for visual diagnosis.
# Uses Seaquest skill checkpoints + env heuristics (no learned meta).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-}:src"
JAX="${PAPER_EVAL_JAXATARI_ROOT:-/home/paul/Documents/JAXAtari-seaquest-robustness}"
OUT="${1:-outputs/videos/paper_eval/rq2_transfer_base}"
MAX_FRAMES="${MAX_FRAMES:-1200}"
SEED="${SEED:-44}"

declare -A CKPT=(
  [ego]="paper_eval/rq1/seaquest_skills_ego/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
  [oc]="paper_eval/rq1/seaquest_skills_oc/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
  [pixel]="paper_eval/rq1/seaquest_skills_pixel/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
)
ENVS=(pong tennis freeway asterix beamrider)

mkdir -p "$OUT"
for mode in ego oc pixel; do
  skills="${CKPT[$mode]}"
  if [[ ! -f "$skills" ]]; then
    echo "SKIP missing ckpt: $skills"
    continue
  fi
  for env in "${ENVS[@]}"; do
    dest="$OUT/$mode"
    mp4="$dest/$env/hierarchical_pqn_seaquest_seed44/base.mp4"
    if [[ -f "$mp4" ]]; then
      echo "EXISTS $mp4"
      continue
    fi
    echo "=== $mode → $env (base, ${MAX_FRAMES} frames) ==="
    python scripts/record_paper_eval_mod_videos.py \
      --skills "$skills" \
      --env "$env" \
      --mods base \
      --max-frames "$MAX_FRAMES" \
      --seed "$SEED" \
      --out-dir "$dest" \
      --jaxatari-root "$JAX" \
      2>&1 | tee "$OUT/${mode}_${env}.log"
  done
done
echo "Done. Videos under $OUT"
