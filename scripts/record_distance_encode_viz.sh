#!/usr/bin/env bash
# Record headless rollout videos for distance_encode_pipeline comparison.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REC="$ROOT/scripts/record_paper_eval_mod_videos.py"
RQ1="$ROOT/outputs/models/paper_eval/rq1"
OUT="$ROOT/outputs/videos/paper_eval/distance_encode_viz"
FRAMES="${1:-500}"
SEED="${2:-44}"

record() {
  local label="$1" env="$2" ckpt="$3"
  local tmp="$OUT/_tmp_${label}"
  echo "=== $label ($env) ==="
  python3 "$REC" \
    --skills "$ckpt" \
    --env "$env" \
    --mods base \
    --max-frames "$FRAMES" \
    --seed "$SEED" \
    --out-dir "$tmp"
  local mp4="$tmp/$env/hierarchical_pqn_seaquest_seed${SEED}/base.mp4"
  mkdir -p "$OUT"
  cp -f "$mp4" "$OUT/${label}.mp4"
  rm -rf "$tmp"
  echo "  -> $OUT/${label}.mp4"
}

# Seaquest — strong / contrasting SQ training runs
record seaquest_clip50_base seaquest \
  "$RQ1/seaquest_skills_ego_clip50_angbins/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
record seaquest_disttier6_base seaquest \
  "$RQ1/seaquest_skills_ego_angbins_disttier6/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
record seaquest_disttier4_base seaquest \
  "$RQ1/seaquest_skills_ego_angbins_disttier4/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
record seaquest_distflags3_base seaquest \
  "$RQ1/seaquest_skills_ego_angbins_distflags3/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"

# Pong — clip50 winner vs repeated failures
record pong_clip50_base pong \
  "$RQ1/seaquest_skills_ego_clip50_angbins/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
record pong_disttier6_base pong \
  "$RQ1/seaquest_skills_ego_angbins_disttier6/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
record pong_disttier4_base pong \
  "$RQ1/seaquest_skills_ego_angbins_disttier4/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
record pong_disttier2_base pong \
  "$RQ1/seaquest_skills_ego_angbins_disttier2/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
record pong_distflags3_base pong \
  "$RQ1/seaquest_skills_ego_angbins_distflags3/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
record pong_distflags6_base pong \
  "$RQ1/seaquest_skills_ego_angbins_distflags6/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"

# Phoenix — tier4 spike vs tier6 stable low
record phoenix_clip50_base phoenix \
  "$RQ1/seaquest_skills_ego_clip50_angbins/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
record phoenix_disttier4_base phoenix \
  "$RQ1/seaquest_skills_ego_angbins_disttier4/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
record phoenix_disttier6_base phoenix \
  "$RQ1/seaquest_skills_ego_angbins_disttier6/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"

echo ""
echo "All videos in $OUT"
