#!/usr/bin/env bash
# Run skill-context ablation train+home-eval pairs back to back on one GPU.
#
#   bash scripts/run_context_ablation_chain.sh GPU STEM [STEM ...]
#
# STEM is a train stem from configs/experiment/paper_eval/context_ablation/,
# e.g. tf_seaquest_skills_ego_abspos_seed44. Its _eval follows directly.
# Optional env: MEM_FRACTION (e.g. 0.30 on cluster2 beside ollama),
# JAXATARI_ROOT (defaults to the Seaquest-robustness fork next to the repo).
#
# Example (detached, one log per GPU):
#   nohup bash scripts/run_context_ablation_chain.sh 1 \
#     tf_seaquest_skills_ego_abspos_seed44 tf_namethisgame_skills_ego_abspos_seed44 \
#     > logs/ctx_abl_gpu1.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 2 ]]; then
  sed -n '2,15p' "$0"; exit 2
fi
GPU="$1"; shift

export JAXATARI_ROOT="${JAXATARI_ROOT:-$(cd "$ROOT/.." && pwd)/JAXAtari-seaquest-robustness}"
export PAPER_EVAL_JAXATARI_ROOT="$JAXATARI_ROOT"
export PYTHONPATH="$JAXATARI_ROOT/src:$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export JAXTARI_AUTO_UPDATE_SPRITES=1
export CBL_EVAL_MAX_PARALLEL_MODS="${CBL_EVAL_MAX_PARALLEL_MODS:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MPLCONFIGDIR="/tmp/mpl-$USER"
export CUDA_VISIBLE_DEVICES="$GPU"
PY="${PY:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
if [[ -n "${MEM_FRACTION:-}" ]]; then
  export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
fi
[[ -d "$JAXATARI_ROOT/src" ]] || { echo "missing $JAXATARI_ROOT/src"; exit 2; }

for stem in "$@"; do
  cfg="configs/experiment/paper_eval/context_ablation/${stem}.yaml"
  [[ -f "$cfg" ]] || { echo "[$(date '+%F %T')] skip: no $cfg"; continue; }
  echo "[$(date '+%F %T')] gpu${GPU} start ${stem} (+ _eval)"
  "$PY" -u scripts/run_paper_eval.py --only "$stem" "${stem}_eval" \
    --wait-deps --skip-rq-gate --lean --resume
  rc=$?
  echo "[$(date '+%F %T')] gpu${GPU} done ${stem} exit=${rc}"
done
