#!/usr/bin/env bash
# Sparse C/D/A clean-story batch — 3 sequential GPU pipelines with wandb videos.
#
# GPU layout (uneven → one suite per GPU, not --gpus round-robin on the union):
#   GPU i: Seaquest 25M sparse → full mod eval → ZS transfers
#   GPU j: Beamrider 10M sparse → Montezuma 10M sparse
#   GPU k: Seaquest pixel tandem 20M sparse → home mod eval
#
# Videos: --video (from-scratch trains only). Do not pass --lean (it disables videos).
#
# Usage:
#   python scripts/materialize_sparse_cda_clean.py
#   bash scripts/run_sparse_cda_clean.sh --gpus 0,1,2
#   bash scripts/run_sparse_cda_clean.sh --gpus 2,3,5   # remap physical ids
#
# Extra args after --gpus are forwarded to each worker (e.g. --resume --wandb-mode online).
#
# Tomorrow (meta-OC ablation, separate batch):
#   python scripts/materialize_seaquest_meta_oc_ablation.py
#   python scripts/run_paper_eval.py --suite seaquest_meta_oc_ablation \
#       --gpus 0,1,2 --wait-deps --lean --skip-rq-gate

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" scripts/materialize_sparse_cda_clean.py

GPUS="0,1,2"
FORWARD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="${2:?}"
      shift 2
      ;;
    --gpus=*)
      GPUS="${1#--gpus=}"
      shift
      ;;
    --lean)
      echo "error: --lean disables wandb videos; omit it for this batch" >&2
      exit 2
      ;;
    *)
      FORWARD+=("$1")
      shift
      ;;
  esac
done

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if [[ ${#GPU_ARR[@]} -ne 3 ]]; then
  echo "error: need exactly 3 GPU ids (got: $GPUS)" >&2
  exit 2
fi

SUITES=(sparse_cda_clean_gpu0 sparse_cda_clean_gpu1 sparse_cda_clean_gpu2)
PIDS=()
for i in 0 1 2; do
  gpu="${GPU_ARR[$i]// /}"
  suite="${SUITES[$i]}"
  echo "=== worker $((i + 1))/3  CUDA_VISIBLE_DEVICES=$gpu  suite=$suite ==="
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/run_paper_eval.py \
    --suite "$suite" \
    --wait-deps \
    --skip-rq-gate \
    --video \
    "${FORWARD[@]}" &
  PIDS+=("$!")
done

worst=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    worst=1
  fi
done
if [[ "$worst" -ne 0 ]]; then
  echo "sparse_cda_clean: one or more workers failed" >&2
  exit "$worst"
fi
echo "sparse_cda_clean: all workers finished successfully."
