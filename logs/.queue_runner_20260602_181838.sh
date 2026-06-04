#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=/home/paul/Documents/ConceptBasedLearning
PYTHON_BIN=/home/paul/.pyenv/versions/cuda_jax/bin/python
EXPERIMENT=seaquest_skills_pretrain_vel_size
GPU_ID=0
WAIT_PID=''
EXTRA_ARGS=('' )

wait_for_slot () 
{ 
    if [[ -n "$WAIT_PID" ]]; then
        echo "$(date -Is): waiting for PID $WAIT_PID to exit";
        while kill -0 "$WAIT_PID" 2> /dev/null; do
            sleep 60;
        done;
        echo "$(date -Is): PID $WAIT_PID finished";
        return;
    fi;
    echo "$(date -Is): waiting for GPU $GPU_ID to have no compute processes";
    while nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader 2> /dev/null | grep -q '[0-9]'; do
        nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2> /dev/null | head -3;
        sleep 60;
    done;
    echo "$(date -Is): GPU $GPU_ID is idle"
}
run_training () 
{ 
    cd "$REPO_ROOT";
    echo "$(date -Is): starting: $PYTHON_BIN main.py experiment=$EXPERIMENT ${EXTRA_ARGS[*]:-}";
    exec "$PYTHON_BIN" main.py "experiment=${EXPERIMENT}" "${EXTRA_ARGS[@]}"
}
wait_for_slot
run_training
