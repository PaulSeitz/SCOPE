#!/usr/bin/env bash
# Sequential Arm1 → Arm2 sparse-reward facing-v2 pipeline (train → SQ eval → Pong).
#
# See tmp/collect_sparse_arms_LOG.md for motivation and gates.
#
# Usage:
#   python scripts/materialize_collect_sparse_arms.py
#   bash scripts/run_collect_sparse_arms.sh --gpus 0 --wait-deps --lean
#
# Extra args are forwarded to run_paper_eval.py.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" scripts/materialize_collect_sparse_arms.py
exec "$PYTHON_BIN" scripts/run_paper_eval.py --suite collect_sparse_arms "$@"
