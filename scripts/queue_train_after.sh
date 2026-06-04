#!/usr/bin/env bash
# Queue a training command to start after a GPU job finishes.
#
# Typical use (background, waits for GPU 0 to be idle):
#   ./scripts/queue_train_after.sh --experiment seaquest_skills_pretrain_vel
#
# Wait for a specific training PID (from pgrep):
#   ./scripts/queue_train_after.sh --wait-pid 22944 --experiment seaquest_skills_pretrain_vel
#
# See: ./scripts/queue_train_after.sh --help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
GPU_ID=0
WAIT_PID=""
FOREGROUND=0
EXTRA_ARGS=()
EXPERIMENT=""

usage() {
  cat <<'EOF'
queue_train_after.sh — run training after the GPU (or a PID) is free

USAGE
  ./scripts/queue_train_after.sh --experiment <name> [options] [-- extra hydra args]

REQUIRED
  --experiment NAME     Hydra experiment (e.g. seaquest_skills_pretrain_vel)

WAIT MODE (pick one; default: --wait-gpu)
  --wait-gpu [ID]         Wait until no compute process uses GPU ID (default: 0)
  --wait-pid PID          Wait until process PID exits

RUN MODE
  (default)               Background via nohup; terminal returns immediately
  --foreground            Block this terminal until training completes

OTHER
  --log FILE              Log file (default: logs/queue_<experiment>_<timestamp>.log)
  --repo PATH             Repo root (auto-detected)
  --python PATH           Python interpreter (default: pyenv which python, else python3)
  -h, --help              Show this help

EXAMPLES
  # A/B: velocity run after current seaquest pretrain exits
  pgrep -af "python main.py experiment=seaquest_skills_pretrain"
  ./scripts/queue_train_after.sh --wait-pid 22944 --experiment seaquest_skills_pretrain_vel

  # Queue when GPU 0 is idle (any job)
  ./scripts/queue_train_after.sh --experiment freeway_fewshot

  # Same seed override + custom log
  ./scripts/queue_train_after.sh --experiment seaquest_skills_pretrain_vel -- SEED=44

MONITOR
  tail -f logs/queue_<experiment>_*.log
  pgrep -af "queue_train_after|main.py experiment="

NOTES
  - Run from a shell where pyenv/venv is already correct, or pass --python explicitly.
  - The waiter inherits your current PATH when you launch the script.
  - Hydra args after -- are forwarded to main.py (e.g. SEED=44 WANDB_MODE=disabled).
EOF
}

resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "$PYTHON_BIN"
    return
  fi
  if command -v pyenv >/dev/null 2>&1; then
    pyenv which python 2>/dev/null && return
  fi
  command -v python3
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --experiment)
      EXPERIMENT="${2:?--experiment requires a value}"
      shift 2
      ;;
    --wait-pid)
      WAIT_PID="${2:?--wait-pid requires a value}"
      shift 2
      ;;
    --wait-gpu)
      if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
        GPU_ID="$2"
        shift 2
      else
        shift 1
      fi
      ;;
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --log)
      LOG_FILE="${2:?--log requires a value}"
      shift 2
      ;;
    --repo)
      REPO_ROOT="${2:?--repo requires a value}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:?--python requires a value}"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$EXPERIMENT" ]]; then
  echo "Error: --experiment is required." >&2
  usage >&2
  exit 1
fi

if [[ -n "$WAIT_PID" ]] && ! kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "Warning: PID $WAIT_PID is not running; starting training immediately." >&2
  WAIT_PID=""
fi

PYTHON_BIN="$(resolve_python)"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/queue_${EXPERIMENT}_${TIMESTAMP}.log}"

_filter_extra_args() {
  local -a filtered=()
  local arg
  for arg in "${EXTRA_ARGS[@]}"; do
    [[ -n "$arg" ]] && filtered+=("$arg")
  done
  EXTRA_ARGS=("${filtered[@]}")
}

run_training() {
  cd "$REPO_ROOT"
  _filter_extra_args
  if ((${#EXTRA_ARGS[@]})); then
    echo "$(date -Is): starting: $PYTHON_BIN main.py experiment=$EXPERIMENT ${EXTRA_ARGS[*]}"
    exec "$PYTHON_BIN" main.py "experiment=${EXPERIMENT}" "${EXTRA_ARGS[@]}"
  else
    echo "$(date -Is): starting: $PYTHON_BIN main.py experiment=$EXPERIMENT"
    exec "$PYTHON_BIN" main.py "experiment=${EXPERIMENT}"
  fi
}

wait_for_slot() {
  if [[ -n "$WAIT_PID" ]]; then
    echo "$(date -Is): waiting for PID $WAIT_PID to exit"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
      sleep 60
    done
    echo "$(date -Is): PID $WAIT_PID finished"
    return
  fi

  echo "$(date -Is): waiting for GPU $GPU_ID to have no compute processes"
  while nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
    nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null | head -3
    sleep 60
  done
  echo "$(date -Is): GPU $GPU_ID is idle"
}

if [[ "$FOREGROUND" -eq 1 ]]; then
  wait_for_slot
  run_training 2>&1 | tee -a "$LOG_FILE"
  exit "${PIPESTATUS[0]}"
fi

RUNNER="${LOG_DIR}/.queue_runner_${TIMESTAMP}.sh"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "REPO_ROOT=$(printf %q "$REPO_ROOT")"
  echo "PYTHON_BIN=$(printf %q "$PYTHON_BIN")"
  echo "EXPERIMENT=$(printf %q "$EXPERIMENT")"
  echo "GPU_ID=$(printf %q "$GPU_ID")"
  echo "WAIT_PID=$(printf %q "$WAIT_PID")"
  _filter_extra_args
  if ((${#EXTRA_ARGS[@]})); then
    echo "EXTRA_ARGS=($(printf '%q ' "${EXTRA_ARGS[@]}"))"
  else
    echo "EXTRA_ARGS=()"
  fi
  echo ""
  declare -f _filter_extra_args wait_for_slot run_training
  echo "wait_for_slot"
  echo "run_training"
} >"$RUNNER"
chmod +x "$RUNNER"

nohup bash "$RUNNER" >>"$LOG_FILE" 2>&1 &
QUEUE_PID=$!
echo "Queued background job PID: $QUEUE_PID"
echo "Runner: $RUNNER"
echo "Log: $LOG_FILE"
echo "Monitor: tail -f $LOG_FILE"
