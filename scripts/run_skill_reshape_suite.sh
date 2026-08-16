#!/usr/bin/env bash
# Thin wrapper → scripts/run_paper_eval.py --suite skill_reshape (RTPT / lean / resume).
#
# Prefer the Python entrypoint on the cluster (pyenv already active):
#   CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py --suite skill_reshape --lean
#   CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py --suite skill_reshape --eval-only
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if command -v pyenv >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-$(pyenv which python 2>/dev/null || true)}"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" scripts/run_paper_eval.py --suite skill_reshape "$@"
