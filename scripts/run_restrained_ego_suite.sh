#!/usr/bin/env bash
# Sequential restrained-ego facing-v2 pipeline (train → SQ eval → Pong).
#
# See docs/restrained_ego_LOG.md.
#
# Usage:
#   python scripts/materialize_restrained_ego_suite.py
#   bash scripts/run_restrained_ego_suite.sh --gpus 0 --wait-deps --lean
#
# Extra args are forwarded to run_paper_eval.py.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" scripts/materialize_restrained_ego_suite.py
exec "$PYTHON_BIN" scripts/run_paper_eval.py --suite restrained_ego "$@"
