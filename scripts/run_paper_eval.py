#!/usr/bin/env python3
"""
Paper evaluation suite runner (RQ1 → RQ2 → RQ3).

Usage:
  python scripts/run_paper_eval.py --list
  python scripts/run_paper_eval.py --list-mods seaquest
  python scripts/run_paper_eval.py --rq 1 --only seaquest_skills_ego
  python scripts/run_paper_eval.py --rq 1 --phase train --shard 1/2
  python scripts/run_paper_eval.py --rq 2   # gated on RQ1 skill ckpts
  python scripts/run_paper_eval.py --smoke --trial   # full wiring smoke test
  python scripts/materialize_paper_eval_configs.py   # regenerate RQ2/RQ3 YAMLs

  # DESTROY ablation (3 GPUs: one variant per shard):
  python scripts/run_paper_eval.py --suite destroy_ablation --gpus 0,1,2 --wait-deps --lean

  # DESTROY urgency (contact+living-tax, no shot ego):
  python scripts/run_paper_eval.py --suite destroy_urgency --wait-deps --lean

  # COLLECT urgency + DESTROY shot-between (2 GPUs, one arm each):
  python scripts/run_paper_eval.py --suite collect_shot_probes --gpus 0,1 --wait-deps --lean

  # Pose-only vs DESTROY-only extras (2 GPUs, one arm each):
  python scripts/run_paper_eval.py --suite ego_pose_probes --gpus 0,1 --wait-deps --lean

  # COLLECT death-off vs DESTROY-first heuristic (2 GPUs, one arm each):
  python scripts/run_paper_eval.py --suite collect_combat_probes --gpus 0,1 --wait-deps --lean

  # Distance clip 50px × 8 overlapping world-θ bins (4 GPUs, one arm each):
  python scripts/run_paper_eval.py --suite ego_clip_bins --gpus 0,1,2,3 --wait-deps --lean

  # Low distance-clip ablations (2 GPUs: clip15 + clip8, then SQ/Pong/Beam/Phoenix):
  python scripts/run_paper_eval.py --suite clip_low_pipeline --gpus 0,1 --wait-deps --lean
  python scripts/run_paper_eval.py --suite clip_sweep_pipeline --gpus 0,1 --wait-deps --lean
  python scripts/run_paper_eval.py --suite distance_encode_pipeline --gpus 0,1 --wait-deps --lean

  # Ground-up thesis Tier A (retrain clip50+θ-bins, full SQ/RQ2/RQ3 pixel tandem):
  bash scripts/prepare_thesis_final_run.sh                  # archive + remove ckpts
  python scripts/run_paper_eval.py --suite thesis_tier_a --gpus 0,1 --wait-deps --lean \\
    --batch-tag thesis_tier_a_$(date +%Y%m%d)
  # Clip100/150 COLLECT-surface probes (Pong distance-tier hypothesis):
  #   python scripts/materialize_clip_collectsurface_probe.py --clip 100
  #   python scripts/materialize_clip_collectsurface_probe.py --clip 150
  #   CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \\
  #     --suite clip100_collectsurface_probe --wait-deps --lean
  #   CUDA_VISIBLE_DEVICES=1 python scripts/run_paper_eval.py \\
  #     --suite clip150_collectsurface_probe --wait-deps --lean
  # Facing distance ablations (direction-only then distflags5; sequential 1 GPU):
  #   python scripts/materialize_facing_distance_ablation.py
  #   CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \\
  #     --suite facing_distance_ablation --wait-deps --lean --resume
  #   python scripts/materialize_thesis_tier_b_facing.py
  #   CUDA_VISIBLE_DEVICES=0,1 python scripts/run_paper_eval.py \\
  #     --suite thesis_tier_b_facing --post-eval --gpus 0,1 --wait-deps --lean --resume
  # Tier C facing (after B): 25M scratch + RQ3 meta/tandem + tandem-scratch BR/FR/AS:
  #   python scripts/materialize_thesis_tier_c_facing.py
  #   CUDA_VISIBLE_DEVICES=0,1 python scripts/run_paper_eval.py \\
  #     --suite thesis_tier_c_facing --post-eval --gpus 0,1 --wait-deps --lean --resume

  # Minimal-engineering ablation — trains only tonight (3 GPUs, ~3h); eval tomorrow:
  python scripts/run_paper_eval.py --suite minimal_ablation_train --gpus 0,1,2 --wait-deps --lean
  python scripts/run_paper_eval.py --suite minimal_ablation --eval-only --resume --gpus 0,1,2,3 --wait-deps --lean

  # Overnight, two GPUs (do not use --gpus on these; pin CUDA yourself):
  #   GPU 0: full eval of existing clip50+θ-bins ckpt, then RQ3 OC tandem
  CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py --suite clip50_angbins_full --wait-deps --lean
  #   GPU 1: retraining with constant SKILL_LR, then the same full eval
  CUDA_VISIBLE_DEVICES=1 python scripts/run_paper_eval.py --suite clip_bins_nodecay --wait-deps --lean

  # POV baseline recovery (3 GPUs: ψ-bins8 / vel+ψ-bins8 / vel+ψ-bins8+turn in parallel):
  python scripts/run_paper_eval.py --suite pov_baseline --gpus 0,1,2 --wait-deps --lean
  python scripts/run_paper_eval.py --suite pov_baseline_train --gpus 0,1,2 --wait-deps --lean
  # Post-facingfix retrain (invalidates pre-2026-08-30 facing-frame ckpts):
  python scripts/run_paper_eval.py --suite pov_facing_v2 --gpus 0,1 --wait-deps --lean
  python scripts/run_paper_eval.py --suite pov_facing_v2_eval --gpus 0,1 --wait-deps --lean
  python scripts/run_paper_eval.py --suite pov_facing_v2_phase2 --gpus 0,1 --wait-deps --lean
  # COLLECT/DESTROY sparse reward arms (sequential; see tmp/collect_sparse_arms_LOG.md):
  bash scripts/run_collect_sparse_arms.sh --gpus 0 --wait-deps --lean
  # Sparse C/D/A clean story (3 GPUs; wandb videos on — do not use --lean):
  #   python scripts/materialize_sparse_cda_clean.py
  #   bash scripts/run_sparse_cda_clean.sh --gpus 0,1,2
  # Meta-OC tandem ablation (tomorrow):
  #   python scripts/materialize_seaquest_meta_oc_ablation.py
  #   python scripts/run_paper_eval.py --suite seaquest_meta_oc_ablation \\
  #       --gpus 0,1,2 --wait-deps --lean --skip-rq-gate
  # Facing v2 breadth (3 tmux sessions — pin GPUs; do not use --gpus here):
  #   GPU0: φ8 zero-shot BR+PH → Phoenix from-scratch φ8
  CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py --suite pov_facing_v2_gpu0 --wait-deps --lean
  #   GPU1: φ8+mid zero-shot BR+PH → Beamrider from-scratch φ8
  CUDA_VISIBLE_DEVICES=1 python scripts/run_paper_eval.py --suite pov_facing_v2_gpu1 --wait-deps --lean
  #   GPU2: Seaquest noclip facing_xy → SQ mods + Pong/PH/BR zero-shot
  CUDA_VISIBLE_DEVICES=2 python scripts/run_paper_eval.py --suite pov_facing_v2_gpu2 --wait-deps --lean

  # Facing / thesis overnight (φ8 clip50 COLLECT-surface ego + POV-mapper pixel/OC):
  #   python scripts/materialize_thesis_facing_suite.py
  #   CUDA_VISIBLE_DEVICES=0,1 python scripts/run_paper_eval.py \\
  #     --suite thesis_facing --gpus 0,1 --wait-deps --lean --resume
  # Surface-gap reward retrains (AABB approach shaping; obs–reward aligned):
  #   2 GPUs — Kangaroo (clip50+θ8) + Seaquest navigatesurface → SQ eval → Pong:
  python scripts/run_paper_eval.py --suite surfacegap_dual --gpus 0,1 --wait-deps --lean
  #   Kangaroo only / Seaquest nav-surfacegap only:
  python scripts/run_paper_eval.py --suite kangaroo_surfacegap --gpus 0 --wait-deps --lean
  python scripts/run_paper_eval.py --suite navigate_surface_surfacegap --gpus 0 --wait-deps --lean
  #   Kangaroo heuristic-fix retrain (POV facing ψ8 surfacegap; mirrors SQ nav-sgap):
  python scripts/run_paper_eval.py --suite kangaroo_facing_surfacegap --gpus 0 --wait-deps --lean
  #   Body-frame gaps: Kangaroo ‖ Seaquest navigatesurface → SQ eval → Pong:
  python scripts/run_paper_eval.py --suite navigate_surface_bodygaps --gpus 0,1 --wait-deps --lean
  #   Ego channel deletion (v2 spine; 5 arms → SQ eval → Pong/BR/PH):
  python scripts/run_paper_eval.py --suite ego_min_channels --gpus 0,1 --wait-deps --lean
  #   Locked spine + DESTROY-only extras retrain:
  python scripts/run_paper_eval.py --suite ego_min_destroyextras --gpus 0 --wait-deps --lean
  #   Locked channels, unclipped distance (one-off):
  python scripts/run_paper_eval.py --suite ego_min_noclip --gpus 0 --wait-deps --lean
  #   Drop can_fire + oval AVOID / surface-sub dodge:
  python scripts/run_paper_eval.py --suite ego_min_noweapon_ovalavoid --gpus 0 --wait-deps --lean
  #   Extra seeds 45/46 for noweapon ovalavoid (YAML SEED; do not pass --seed):
  python scripts/run_paper_eval.py --suite ego_min_noweapon_ovalavoid_seeds --gpus 0,1 --wait-deps --lean
  #   Heuristic factor ablation (oval-only vs isotropic+surface-sub5):
  python scripts/run_paper_eval.py --suite ego_min_noweapon_heu_ablation --gpus 0,1 --wait-deps --lean
  #   Plan B: can_fire on (noblocking_nofacing) + same oval heuristic:
  python scripts/run_paper_eval.py --suite ego_min_nofacing_ovalavoid --gpus 0 --wait-deps --lean
  #   Clip sweep on noweapon+surfacesub5 (10/20/30/50/80/100/noclip; --resume skips clip50):
  python scripts/run_paper_eval.py --suite ego_min_noweapon_surfacesub5_clipsweep --gpus 0,1,2 --wait-deps --lean --resume
  #   Ungated skill-mechanics reward check (clip10+clip50 + clip10 all-objects):
  python scripts/materialize_surfacesub5_skillmech_rw.py
  python scripts/run_paper_eval.py --suite ego_min_surfacesub5_skillmech_rw --gpus 0,1,2 --wait-deps --lean
  #   Same-scaffolding fixed ego-rule baseline (angle servo / constant actions):
  python scripts/materialize_fixed_ego_skill_baseline.py
  python scripts/run_paper_eval.py --suite fixed_ego_skill_baseline --gpus 0 --lean
  #   Exclusive near/mid/far (12/40), no sticky:
  python scripts/materialize_surfacesub5_distbands.py
  python scripts/run_paper_eval.py --suite ego_min_noweapon_surfacesub5_distbands --gpus 0 --wait-deps --lean
  #   Clip50 + Machado sticky 0.25 (train only; eval still sticky=0):
  python scripts/run_paper_eval.py --suite ego_min_noweapon_surfacesub5_clip50_sticky --gpus 1 --wait-deps --lean
  #   Distance encoding + sticky on surfacesub5 (clip0 / distbins8 / clip10+sticky):
  python scripts/materialize_surfacesub5_distabl.py
  python scripts/run_paper_eval.py --suite ego_min_noweapon_surfacesub5_distabl --gpus 0,1,2 --wait-deps --lean

  # SKILL_TARGET_CUT=lifetime ablations (sequential on one GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py --suite kangaroo_lifetime --wait-deps --lean \\
    && CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py --suite clip50_lifetime_pipeline --wait-deps --lean

  # Skill-reshape validation (COLLECT approach + AVOID escape; cluster RTPT):
  CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py --suite skill_reshape
  CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py --suite skill_reshape --eval-only
  CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py --suite skill_reshape --trial --lean

  # One-shot overnight on two GPUs (workers poll for RQ1 ckpts before gated jobs):
  python scripts/run_paper_eval.py --gpus 0,1 --wait-deps

Fail-fast guards (avoid burning hours / idling forever):
  - Abort if free disk on the project volume drops below --min-free-gb (default 2).
  - Treat ENOSPC / "No space left on device" as fatal for the whole suite.
  - With --wait-deps, do not wait for stems this worker already failed to produce.

Disk-lean overnight (shared /home nearly full):
  python scripts/run_paper_eval.py --gpus 0,1 --wait-deps --lean
  # --lean: no videos, no training_metrics.csv, wandb local dirs under $TMPDIR

Resume remaining work on 3 GPUs (skip finished trains+evals; drop cancelled pixel RQ1):
  python scripts/run_paper_eval.py --gpus 0,1,2 --wait-deps --lean --resume \\
    --exclude seaquest_skills_pixel seaquest_skills_pixel_eval

  # Preview what is still missing (includes incomplete/missing *_mods.json evals):
  python scripts/run_paper_eval.py --resume --list \\
    --exclude seaquest_skills_pixel seaquest_skills_pixel_eval
"""

from __future__ import annotations

import argparse
import copy
import errno
import gc
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# Abort when free space on the project volume drops below this.
# Modest default: shared /home is often nearly full; ckpts << videos/wandb.
DEFAULT_MIN_FREE_GB = 2.0

# Avoid XLA:GPU Triton GEMM fusion CHECK failures
# (triton_tiling_propagation.cc: dst_remaining_size == 1) seen on some
# jaxlib/CUDA stacks during first compile of large train graphs.
_DEFAULT_XLA_FLAGS = (
    "--xla_gpu_strict_conv_algorithm_picker=false "
    "--xla_gpu_enable_triton_gemm=false"
)
_existing = os.environ.get("XLA_FLAGS", "").strip()
if not _existing:
    os.environ["XLA_FLAGS"] = _DEFAULT_XLA_FLAGS
elif "--xla_gpu_enable_triton_gemm" not in _existing:
    os.environ["XLA_FLAGS"] = f"{_existing} --xla_gpu_enable_triton_gemm=false"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_JAXATARI_FORK = Path("/home/paul/Documents/JAXAtari-seaquest-robustness")
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "paper_eval"
MODS_DIR = CONFIG_DIR / "mods"
RTPT_BASE = "HRL"
DEFAULT_SEED = 44
TRIAL_STEPS_DEFAULT = 16384  # one update at 128×128
# Full-game horizon for all paper_eval mod/transfer tables (trial/smoke stay short).
SUITE_EVAL_MAX_FRAMES = 27000

# After COLLECT approach + AVOID escape reshaping: train ego Seaquest, then
# standard mod eval, Pong transfer, BeamRider (idle FIRE).
SKILL_RESHAPE_SUITE = "skill_reshape"
SKILL_RESHAPE_STEMS = (
    "seaquest_skills_ego",
    "seaquest_skills_ego_eval",
    "pong_transfer_ego",
    "beamrider_transfer_ego",
)
SKILL_RESHAPE_EVAL_STEMS = tuple(
    s for s in SKILL_RESHAPE_STEMS if s != "seaquest_skills_ego"
)
DESTROY_ABLATION_SUITE = "destroy_ablation"
# Order is shard-critical: 3 GPUs get (train, eval, pong) of one variant each.
DESTROY_ABLATION_STEMS = (
    "seaquest_skills_ego_destroy_yalign",
    "seaquest_skills_ego_destroy_standoff",
    "seaquest_skills_ego_destroy_contact",
    "seaquest_skills_ego_destroy_yalign_eval",
    "seaquest_skills_ego_destroy_standoff_eval",
    "seaquest_skills_ego_destroy_contact_eval",
    "pong_transfer_ego_destroy_yalign",
    "pong_transfer_ego_destroy_standoff",
    "pong_transfer_ego_destroy_contact",
)
DESTROY_URGENCY_SUITE = "destroy_urgency"
# Shot-ego arm dropped: it underperformed; suite is urgency (reward+heuristic) only.
DESTROY_URGENCY_STEMS = (
    "seaquest_skills_ego_destroy_urgency",
    "seaquest_skills_ego_destroy_urgency_eval",
    "pong_transfer_ego_destroy_urgency",
)
COLLECT_SHOT_PROBES_SUITE = "collect_shot_probes"
# Shard-critical for 2 GPUs: GPU i gets train/eval/pong of one arm.
COLLECT_SHOT_PROBES_STEMS = (
    "seaquest_skills_ego_collect_urgency",
    "seaquest_skills_ego_destroy_shot_between",
    "seaquest_skills_ego_collect_urgency_eval",
    "seaquest_skills_ego_destroy_shot_between_eval",
    "pong_transfer_ego_collect_urgency",
    "pong_transfer_ego_destroy_shot_between",
)
EGO_POSE_PROBES_SUITE = "ego_pose_probes"
# Pose-only (12-dim all heads) vs DESTROY-only extras (DESTROY 24 / others 12).
EGO_POSE_PROBES_STEMS = (
    "seaquest_skills_ego_pose",
    "seaquest_skills_ego_destroy_extras",
    "seaquest_skills_ego_pose_eval",
    "seaquest_skills_ego_destroy_extras_eval",
    "pong_transfer_ego_pose",
    "pong_transfer_ego_destroy_extras",
)
COLLECT_COMBAT_PROBES_SUITE = "collect_combat_probes"
# Isolated urg-v2 clones: COLLECT death penalty off vs DESTROY-first heuristic.
COLLECT_COMBAT_PROBES_STEMS = (
    "seaquest_skills_ego_collect_nodeath",
    "seaquest_skills_ego_destroy_first",
    "seaquest_skills_ego_collect_nodeath_eval",
    "seaquest_skills_ego_destroy_first_eval",
    "pong_transfer_ego_collect_nodeath",
    "pong_transfer_ego_destroy_first",
)
NO_SURFACE_DEATH_SUITE = "no_surface_death"
NO_SURFACE_DEATH_STEMS = (
    "seaquest_skills_ego_no_surface_death",
    "seaquest_skills_ego_no_surface_death_eval",
    "pong_transfer_ego_no_surface_death",
)
# urg-v2 + new heuristic: 2×2 of distance clip 50px and 8 world-θ bins.
# Shard-critical for 4 GPUs: GPU i gets train/eval/pong of one arm.
EGO_CLIP_BINS_SUITE = "ego_clip_bins"
EGO_CLIP_BINS_STEMS = (
    "seaquest_skills_ego_clip_bins_base",
    "seaquest_skills_ego_clip50",
    "seaquest_skills_ego_angbins",
    "seaquest_skills_ego_clip50_angbins",
    "seaquest_skills_ego_clip_bins_base_eval",
    "seaquest_skills_ego_clip50_eval",
    "seaquest_skills_ego_angbins_eval",
    "seaquest_skills_ego_clip50_angbins_eval",
    "pong_transfer_ego_clip_bins_base",
    "pong_transfer_ego_clip50",
    "pong_transfer_ego_angbins",
    "pong_transfer_ego_clip50_angbins",
)
CLIP_BINS_NODECAY_SUITE = "clip_bins_nodecay"
CLIP_BINS_NODECAY_STEMS = (
    "seaquest_skills_ego_clip50_angbins_nodecay",
    "seaquest_skills_ego_clip50_angbins_nodecay_eval",
    "pong_transfer_ego_clip50_angbins_nodecay",
    "tennis_transfer_ego_clip50_angbins_nodecay",
    "beamrider_transfer_ego_clip50_angbins_nodecay",
)
# Paper-scale eval of the clip50+θ-bins winner (existing ckpt). 1 GPU:
# full Seaquest/Pong/Tennis/BeamRider tables, then RQ3 OC tandem.
CLIP50_ANGBINS_FULL_SUITE = "clip50_angbins_full"
CLIP50_ANGBINS_FULL_STEMS = (
    "seaquest_skills_ego_clip50_angbins_eval_full",
    "pong_transfer_ego_clip50_angbins_full",
    "tennis_transfer_ego_clip50_angbins",
    "beamrider_transfer_ego_clip50_angbins",
    "seaquest_tandem_oc_ego_clip50_angbins",
    "seaquest_tandem_oc_ego_clip50_angbins_eval",
)
# Ground-up thesis Tier A: RQ1 all arms, RQ2 ego transfer (6 envs), RQ3 pixel tandem.
THESIS_TIER_A_SUITE = "thesis_tier_a"
THESIS_TIER_A_STEMS = (
    # --- RQ1 trains (clip50 ego gates transfers / RQ3) ---
    "seaquest_skills_ego_clip50_angbins",
    "seaquest_skills_pixel",
    "seaquest_skills_oc",
    "seaquest_pqn",
    "seaquest_pqn_oc",
    # --- RQ1 eval (full Seaquest mod allowlist @ 27k) ---
    "seaquest_skills_ego_clip50_angbins_eval_full",
    "seaquest_skills_pixel_eval",
    "seaquest_skills_oc_eval",
    "seaquest_pqn_eval",
    "seaquest_pqn_oc_eval",
    "seaquest_heuristic_random",
    # --- RQ2 zero-shot transfer (ego / clip50 winner) ---
    "pong_transfer_ego",
    "tennis_transfer_ego",
    "beamrider_transfer_ego",
    "freeway_transfer_ego",
    "asterix_transfer_ego",
    "phoenix_transfer_ego",
    # --- RQ3 pixel tandem ---
    "seaquest_tandem_pixel_ego_clip50_angbins",
    "seaquest_tandem_pixel_ego_clip50_angbins_eval",
)

# Overnight paper spine: facing φ8 clip50 COLLECT-surface ego + POV-mapper pixel/OC.
# Ego/pixel/OC skill trains are fresh stems (surface/goal relabel + mapper lockout).
# --resume only skips existing seaquest_pqn / pqn_oc / heuristic ckpts.
THESIS_FACING_SUITE = "thesis_facing"
THESIS_FACING_STEMS = (
    # --- RQ1 trains ---
    "seaquest_skills_ego_pov_clip50_facing_psibins8_collectsurface",
    "seaquest_skills_pixel_pov",
    "seaquest_skills_oc_pov",
    "seaquest_pqn",
    "seaquest_pqn_oc",
    # --- RQ1 evals ---
    "seaquest_skills_ego_pov_clip50_facing_psibins8_collectsurface_eval_full",
    "seaquest_skills_pixel_pov_eval_full",
    "seaquest_skills_oc_pov_eval_full",
    "seaquest_skills_pixel_pov_eval_nooutline",
    "seaquest_pqn_eval",
    "seaquest_pqn_oc_eval",
    "seaquest_heuristic_random",
    # --- RQ2 ego zero-shot ---
    "pong_transfer_ego_pov_clip50_facing_psibins8_collectsurface",
    "tennis_transfer_ego_pov_clip50_facing_psibins8_collectsurface",
    "beamrider_transfer_ego_pov_clip50_facing_psibins8_collectsurface",
    "freeway_transfer_ego_pov_clip50_facing_psibins8_collectsurface",
    "asterix_transfer_ego_pov_clip50_facing_psibins8_collectsurface",
    "phoenix_transfer_ego_pov_clip50_facing_psibins8_collectsurface",
    # --- RQ2 pixel / OC zero-shot (same facing-relative mapper) ---
    "pong_transfer_pixel_pov",
    "tennis_transfer_pixel_pov",
    "beamrider_transfer_pixel_pov",
    "freeway_transfer_pixel_pov",
    "asterix_transfer_pixel_pov",
    "phoenix_transfer_pixel_pov",
    "pong_transfer_oc_pov",
    "tennis_transfer_oc_pov",
    "beamrider_transfer_oc_pov",
    "freeway_transfer_oc_pov",
    "asterix_transfer_oc_pov",
    "phoenix_transfer_oc_pov",
)

# Tier B: 5M finetune + post-eval on all RQ2 envs × ego/pixel/oc.
THESIS_TIER_B_SUITE = "thesis_tier_b"
THESIS_TIER_B_ENVS = ("pong", "tennis", "beamrider", "freeway", "asterix", "phoenix")
THESIS_TIER_B_MODES = ("ego", "pixel", "oc")
THESIS_TIER_B_STEMS = tuple(
    f"{env}_finetune_{mode}" for env in THESIS_TIER_B_ENVS for mode in THESIS_TIER_B_MODES
)

# Tier B facing: 5M finetune from thesis_facing RQ1 ckpts (POV mapper + φ8 ego).
# Materialize: python scripts/materialize_thesis_tier_b_facing.py
THESIS_TIER_B_FACING_SUITE = "thesis_tier_b_facing"
THESIS_TIER_B_FACING_STEMS = tuple(
    f"{env}_finetune_ego_pov_collectsurface" if mode == "ego" else f"{env}_finetune_{mode}_pov"
    for env in THESIS_TIER_B_ENVS
    for mode in THESIS_TIER_B_MODES
)

# Clip100 distance probe: retrain COLLECT-surface φ8 + Pong zero-shot (Pong ~1px hypothesis).
CLIP100_COLLECTSURFACE_PROBE_SUITE = "clip100_collectsurface_probe"
CLIP100_COLLECTSURFACE_PROBE_STEMS = (
    "seaquest_skills_ego_pov_clip100_facing_psibins8_collectsurface",
    "seaquest_skills_ego_pov_clip100_facing_psibins8_collectsurface_eval_full",
    "pong_transfer_ego_pov_clip100_facing_psibins8_collectsurface",
)

CLIP150_COLLECTSURFACE_PROBE_SUITE = "clip150_collectsurface_probe"
CLIP150_COLLECTSURFACE_PROBE_STEMS = (
    "seaquest_skills_ego_pov_clip150_facing_psibins8_collectsurface",
    "seaquest_skills_ego_pov_clip150_facing_psibins8_collectsurface_eval_full",
    "pong_transfer_ego_pov_clip150_facing_psibins8_collectsurface",
)

# Facing φ8 COLLECT-surface: direction-only (clip0) then 5 nested distance tiers.
# Order is execution order: each arm train → eval_full → pong → beamrider (1 GPU sequential).
# Materialize: python scripts/materialize_facing_distance_ablation.py
FACING_DISTANCE_ABLATION_SUITE = "facing_distance_ablation"
FACING_DISTANCE_ABLATION_STEMS = (
    # --- nodist (clip0) ---
    "seaquest_skills_ego_pov_clip0_facing_psibins8_collectsurface",
    "seaquest_skills_ego_pov_clip0_facing_psibins8_collectsurface_eval_full",
    "pong_transfer_ego_pov_clip0_facing_psibins8_collectsurface",
    "beamrider_transfer_ego_pov_clip0_facing_psibins8_collectsurface",
    # --- distflags5 [15, 25, 50, 100, 200] ---
    "seaquest_skills_ego_pov_facing_psibins8_collectsurface_distflags5",
    "seaquest_skills_ego_pov_facing_psibins8_collectsurface_distflags5_eval_full",
    "pong_transfer_ego_pov_facing_psibins8_collectsurface_distflags5",
    "beamrider_transfer_ego_pov_facing_psibins8_collectsurface_distflags5",
)

# Tier C: 25M from-scratch + post-eval (ego/pixel) + RQ3 meta/tandem extras.
THESIS_TIER_C_SUITE = "thesis_tier_c"
THESIS_TIER_C_STEMS = tuple(
    f"{env}_from_scratch_{mode}"
    for env in THESIS_TIER_B_ENVS
    for mode in ("ego", "pixel")
) + (
    "seaquest_meta_pixel_ego_skills_clip50",
    "seaquest_meta_pixel_ego_skills_clip50_eval",
    "seaquest_meta_oc_ego_skills_clip50",
    "seaquest_meta_oc_ego_skills_clip50_eval",
    "seaquest_tandem_oc_ego_clip50_angbins",
    "seaquest_tandem_oc_ego_clip50_angbins_eval",
)

# Tier C facing: POV mapper + facing φ8 ego; RQ3 meta/tandem from collectsurface ckpt.
# Materialize: python scripts/materialize_thesis_tier_c_facing.py
THESIS_TIER_C_FACING_SUITE = "thesis_tier_c_facing"
_TIER_C_TANDEM_SCRATCH_ENVS = ("beamrider", "freeway", "asterix")
THESIS_TIER_C_FACING_STEMS = tuple(
    f"{env}_from_scratch_ego_pov_collectsurface" for env in THESIS_TIER_B_ENVS
) + tuple(
    f"{env}_from_scratch_pixel_pov" for env in THESIS_TIER_B_ENVS
) + (
    "seaquest_meta_pixel_ego_pov",
    "seaquest_meta_pixel_ego_pov_eval",
    "seaquest_meta_oc_ego_pov",
    "seaquest_meta_oc_ego_pov_eval",
    "seaquest_tandem_oc_ego_pov",
    "seaquest_tandem_oc_ego_pov_eval",
    "seaquest_tandem_pixel_ego_pov",
    "seaquest_tandem_pixel_ego_pov_eval",
) + tuple(
    f"{env}_from_scratch_tandem_{meta}_ego_pov"
    for env in _TIER_C_TANDEM_SCRATCH_ENVS
    for meta in ("pixel", "oc")
)

# 2-GPU batch: NAVIGATE-surface spine A/B + missing pixel tandem (independent trains).
# Even idx → GPU0 (navigate spine → SQ eval → Pong ZS); odd → GPU1 (pixel tandem → eval).
NAVIGATE_SURFACE_TANDEM_PIXEL_SUITE = "navigate_surface_tandem_pixel"
NAVIGATE_SURFACE_TANDEM_PIXEL_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface",
    "seaquest_tandem_pixel_ego_pov",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface_eval_full",
    "seaquest_tandem_pixel_ego_pov_eval",
    "pong_transfer_ego_pov_clip50_facing_psibins8_navigatesurface",
)

# NAVIGATE-surface retrain with AABB surface-gap approach shaping (obs–reward aligned).
# Requires skill_rewards._approach_distances (post-2026-09-05). Sequential 1 GPU.
NAVIGATE_SURFACE_SURFACEGAP_SUITE = "navigate_surface_surfacegap"
NAVIGATE_SURFACE_SURFACEGAP_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface_surfacegap",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface_surfacegap_eval_full",
    "pong_transfer_ego_pov_clip50_facing_psibins8_navigatesurface_surfacegap",
)

# Kangaroo retrain: Seaquest-proven clip50+θ8 angbins + surface-gap rewards.
KANGAROO_SURFACEGAP_SUITE = "kangaroo_surfacegap"
KANGAROO_SURFACEGAP_STEMS = (
    "kangaroo_skills_ego_clip50_angbins_surfacegap",
)

# Kangaroo retrain after heuristic coco/fruit fixes — mirrors Seaquest
# navigatesurface_surfacegap (POV facing, ψ8, clip50; not centres geom).
#   python scripts/run_paper_eval.py --suite kangaroo_facing_surfacegap --gpus 0 --wait-deps --lean
KANGAROO_FACING_SURFACEGAP_SUITE = "kangaroo_facing_surfacegap"
KANGAROO_FACING_SURFACEGAP_STEMS = (
    "kangaroo_skills_ego_pov_clip50_facing_psibins8_surfacegap",
)

# 2-GPU dual: shard-critical — trains first (GPU0 kangaroo, GPU1 Seaquest nav-surfacegap).
#   python scripts/run_paper_eval.py --suite surfacegap_dual --gpus 0,1 --wait-deps --lean
SURFACEGAP_DUAL_SUITE = "surfacegap_dual"
SURFACEGAP_DUAL_STEMS = (
    "kangaroo_skills_ego_clip50_angbins_surfacegap",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface_surfacegap",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface_surfacegap_eval_full",
    "pong_transfer_ego_pov_clip50_facing_psibins8_navigatesurface_surfacegap",
)

# 2-GPU: hit-corridor vs closest-points φ (both on navigatesurface_surfacegap spine).
#   python scripts/run_paper_eval.py --suite surfacegap_geom_dual --gpus 0,1 --wait-deps --lean
SURFACEGAP_GEOM_DUAL_SUITE = "surfacegap_geom_dual"
SURFACEGAP_GEOM_DUAL_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface_surfacegap_hitcorridor",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface_surfacegap_closestphi",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface_surfacegap_hitcorridor_eval_full",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface_surfacegap_closestphi_eval_full",
    "pong_transfer_ego_pov_clip50_facing_psibins8_navigatesurface_surfacegap_hitcorridor",
    "pong_transfer_ego_pov_clip50_facing_psibins8_navigatesurface_surfacegap_closestphi",
)

# Fully centre-consistent navigatesurface (obs+reward+φ). 1 GPU sequential.
#   python scripts/run_paper_eval.py --suite navigate_surface_centres --gpus 0 --wait-deps --lean
NAVIGATE_SURFACE_CENTRES_SUITE = "navigate_surface_centres"
NAVIGATE_SURFACE_CENTRES_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface_centres",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface_centres_eval_full",
    "pong_transfer_ego_pov_clip50_facing_psibins8_navigatesurface_centres",
)

# Body-frame signed surface gaps (obs+reward+φ). 2-GPU dual — shard-critical:
# trains first (GPU0 kangaroo, GPU1 Seaquest train→eval→Pong).
#   python scripts/run_paper_eval.py --suite navigate_surface_bodygaps --gpus 0,1 --wait-deps --lean
NAVIGATE_SURFACE_BODYGAPS_SUITE = "navigate_surface_bodygaps"
NAVIGATE_SURFACE_BODYGAPS_STEMS = (
    "kangaroo_skills_ego_pov_clip50_facing_psibins8_bodygaps",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface_bodygaps",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface_bodygaps_eval_full",
    "pong_transfer_ego_pov_clip50_facing_psibins8_navigatesurface_bodygaps",
)

# EVAL_PLAN §0 — channel-deletion ablations on v2 φ8 clip50 facing (25M).
# 2-GPU shard-critical: all 5 trains first (GPU0: 3 arms, GPU1: 2), then
# Seaquest full eval + Pong / Beamrider / Phoenix ZS per arm.
#   python scripts/run_paper_eval.py --suite ego_min_channels --gpus 0,1 --wait-deps --lean
EGO_MIN_CHANNELS_SUITE = "ego_min_channels"
_EGO_MIN_CHANNEL_ARMS = (
    "noblocking",
    "noblocking_nofacing",
    "noblocking_noweapon",
    "nofacing",
    "noweapon",
)
EGO_MIN_CHANNELS_STEMS = (
    tuple(
        f"seaquest_skills_ego_pov_clip50_facing_psibins8_v2_{a}"
        for a in _EGO_MIN_CHANNEL_ARMS
    )
    + tuple(
        f"seaquest_skills_ego_pov_clip50_facing_psibins8_v2_{a}_eval_full"
        for a in _EGO_MIN_CHANNEL_ARMS
    )
    + tuple(
        f"pong_transfer_ego_pov_clip50_facing_psibins8_v2_{a}"
        for a in _EGO_MIN_CHANNEL_ARMS
    )
    + tuple(
        f"beamrider_transfer_ego_pov_clip50_facing_psibins8_v2_{a}"
        for a in _EGO_MIN_CHANNEL_ARMS
    )
    + tuple(
        f"phoenix_transfer_ego_pov_clip50_facing_psibins8_v2_{a}"
        for a in _EGO_MIN_CHANNEL_ARMS
    )
)

# Locked spine (noblocking_nofacing) + USE_DESTROY_ONLY_EGO_EXTRAS=true retrain.
# COLLECT/AVOID/NAVIGATE get pose-only inputs; DESTROY keeps can_fire.
#   python scripts/run_paper_eval.py --suite ego_min_destroyextras --gpus 0 --wait-deps --lean
EGO_MIN_DESTROYEXTRAS_SUITE = "ego_min_destroyextras"
EGO_MIN_DESTROYEXTRAS_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_nofacing_destroyextras",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_nofacing_destroyextras_eval_full",
    "pong_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_nofacing_destroyextras",
    "beamrider_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_nofacing_destroyextras",
    "phoenix_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_nofacing_destroyextras",
)

# One-off: locked noblocking_nofacing channels, but EGO_DISTANCE_CLIP_MAX_PX=null.
#   python scripts/run_paper_eval.py --suite ego_min_noclip --gpus 0 --wait-deps --lean
EGO_MIN_NOCLIP_SUITE = "ego_min_noclip"
EGO_MIN_NOCLIP_STEMS = (
    "seaquest_skills_ego_pov_noclip_facing_psibins8_v2_noblocking_nofacing",
    "seaquest_skills_ego_pov_noclip_facing_psibins8_v2_noblocking_nofacing_eval_full",
    "pong_transfer_ego_pov_noclip_facing_psibins8_v2_noblocking_nofacing",
    "beamrider_transfer_ego_pov_noclip_facing_psibins8_v2_noblocking_nofacing",
    "phoenix_transfer_ego_pov_noclip_facing_psibins8_v2_noblocking_nofacing",
)

# Drop can_fire (noblocking_noweapon) + oval AVOID interrupt + anytime surface-sub dodge.
#   python scripts/run_paper_eval.py --suite ego_min_noweapon_ovalavoid --gpus 0 --wait-deps --lean
EGO_MIN_NOWEAPON_OVALAVOID_SUITE = "ego_min_noweapon_ovalavoid"
EGO_MIN_NOWEAPON_OVALAVOID_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_ovalavoid",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_ovalavoid_eval_full",
    "pong_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_ovalavoid",
    "beamrider_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_ovalavoid",
    "phoenix_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_ovalavoid",
)

# Extra seeds for noweapon ovalavoid (YAML SEED=45/46; do not pass --seed).
#   python scripts/run_paper_eval.py --suite ego_min_noweapon_ovalavoid_seeds --gpus 0,1 --wait-deps --lean
EGO_MIN_NOWEAPON_OVALAVOID_SEEDS_SUITE = "ego_min_noweapon_ovalavoid_seeds"
_EGO_OVA_SEEDS = (45, 46)
EGO_MIN_NOWEAPON_OVALAVOID_SEEDS_STEMS = (
    tuple(
        f"seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_ovalavoid_seed{seed}"
        for seed in _EGO_OVA_SEEDS
    )
    + tuple(
        f"seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_ovalavoid_seed{seed}_eval_full"
        for seed in _EGO_OVA_SEEDS
    )
    + tuple(
        f"pong_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_ovalavoid_seed{seed}"
        for seed in _EGO_OVA_SEEDS
    )
    + tuple(
        f"beamrider_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_ovalavoid_seed{seed}"
        for seed in _EGO_OVA_SEEDS
    )
    + tuple(
        f"phoenix_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_ovalavoid_seed{seed}"
        for seed in _EGO_OVA_SEEDS
    )
)

# Heuristic factor ablation (noweapon): oval-missile-only vs isotropic+surface-sub5.
#   python scripts/run_paper_eval.py --suite ego_min_noweapon_heu_ablation --gpus 0,1 --wait-deps --lean
EGO_MIN_NOWEAPON_HEU_ABLATION_SUITE = "ego_min_noweapon_heu_ablation"
_EGO_HEU_ABLATION_ARMS = ("oval_only", "surfacesub5")
EGO_MIN_NOWEAPON_HEU_ABLATION_STEMS = (
    tuple(
        f"seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_{a}"
        for a in _EGO_HEU_ABLATION_ARMS
    )
    + tuple(
        f"seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_{a}_eval_full"
        for a in _EGO_HEU_ABLATION_ARMS
    )
    + tuple(
        f"pong_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_{a}"
        for a in _EGO_HEU_ABLATION_ARMS
    )
    + tuple(
        f"beamrider_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_{a}"
        for a in _EGO_HEU_ABLATION_ARMS
    )
    + tuple(
        f"phoenix_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_{a}"
        for a in _EGO_HEU_ABLATION_ARMS
    )
)

# Plan B: locked noblocking_nofacing (can_fire on) + same oval/surface-sub heuristic.
#   python scripts/run_paper_eval.py --suite ego_min_nofacing_ovalavoid --gpus 0 --wait-deps --lean
EGO_MIN_NOFACING_OVALAVOID_SUITE = "ego_min_nofacing_ovalavoid"
EGO_MIN_NOFACING_OVALAVOID_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_nofacing_ovalavoid",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_nofacing_ovalavoid_eval_full",
    "pong_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_nofacing_ovalavoid",
    "beamrider_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_nofacing_ovalavoid",
    "phoenix_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_nofacing_ovalavoid",
)

# Distance-clip sweep on noweapon+surfacesub5 spine (isotropic AVOID 55×55 + surf-sub 5×5).
# Arms: 10/20/30/50/80/100/noclip → SQ full + Pong/BR/PH. clip50 reuses existing heu-ablation ckpt.
#   python scripts/run_paper_eval.py --suite ego_min_noweapon_surfacesub5_clipsweep --gpus 0,1,2 --wait-deps --lean --resume
EGO_MIN_NOWEAPON_SURFACESUB5_CLIPSWEEP_SUITE = "ego_min_noweapon_surfacesub5_clipsweep"
# Alias for the old ovalavoid clipsweep name (retargeted).
EGO_MIN_NOWEAPON_OVALAVOID_CLIPSWEEP_SUITE = EGO_MIN_NOWEAPON_SURFACESUB5_CLIPSWEEP_SUITE
_EGO_SS5_CLIP_TOKS = ("clip10", "clip20", "clip30", "clip50", "clip80", "clip100", "noclip")
EGO_MIN_NOWEAPON_SURFACESUB5_CLIPSWEEP_STEMS = (
    tuple(
        f"seaquest_skills_ego_pov_{tok}_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
        for tok in _EGO_SS5_CLIP_TOKS
    )
    + tuple(
        f"seaquest_skills_ego_pov_{tok}_facing_psibins8_v2_noblocking_noweapon_surfacesub5_eval_full"
        for tok in _EGO_SS5_CLIP_TOKS
    )
    + tuple(
        f"pong_transfer_ego_pov_{tok}_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
        for tok in _EGO_SS5_CLIP_TOKS
    )
    + tuple(
        f"beamrider_transfer_ego_pov_{tok}_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
        for tok in _EGO_SS5_CLIP_TOKS
    )
    + tuple(
        f"phoenix_transfer_ego_pov_{tok}_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
        for tok in _EGO_SS5_CLIP_TOKS
    )
)
EGO_MIN_NOWEAPON_OVALAVOID_CLIPSWEEP_STEMS = EGO_MIN_NOWEAPON_SURFACESUB5_CLIPSWEEP_STEMS

# clip0 (distance zeroed) / distbins8 (no hard clip) / clip10+sticky train.
#   python scripts/materialize_surfacesub5_distabl.py
#   python scripts/run_paper_eval.py --suite ego_min_noweapon_surfacesub5_distabl --gpus 0,1,2 --wait-deps --lean
EGO_MIN_NOWEAPON_SURFACESUB5_DISTABL_SUITE = "ego_min_noweapon_surfacesub5_distabl"
_EGO_SS5_DISTABL_TOKS = ("clip0", "distbins8", "clip10")
_EGO_SS5_DISTABL_SUFFIX = {
    "clip0": "",
    "distbins8": "",
    "clip10": "_sticky",
}


def _ego_ss5_distabl_train_stem(tok: str) -> str:
    return (
        f"seaquest_skills_ego_pov_{tok}_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
        f"{_EGO_SS5_DISTABL_SUFFIX[tok]}"
    )


EGO_MIN_NOWEAPON_SURFACESUB5_DISTABL_STEMS = (
    tuple(_ego_ss5_distabl_train_stem(tok) for tok in _EGO_SS5_DISTABL_TOKS)
    + tuple(f"{_ego_ss5_distabl_train_stem(tok)}_eval_full" for tok in _EGO_SS5_DISTABL_TOKS)
    + tuple(
        f"{env}_transfer_ego_pov_{tok}_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
        f"{_EGO_SS5_DISTABL_SUFFIX[tok]}"
        for env in ("pong", "beamrider", "phoenix")
        for tok in _EGO_SS5_DISTABL_TOKS
    )
)

# Exclusive near/mid/far (12px / 40px), no sticky.
#   python scripts/materialize_surfacesub5_distbands.py
#   python scripts/run_paper_eval.py --suite ego_min_noweapon_surfacesub5_distbands --gpus 0 --wait-deps --lean
EGO_MIN_NOWEAPON_SURFACESUB5_DISTBANDS_SUITE = "ego_min_noweapon_surfacesub5_distbands"
_EGO_SS5_DISTBANDS_TRAIN = (
    "seaquest_skills_ego_pov_distbands3_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
)
EGO_MIN_NOWEAPON_SURFACESUB5_DISTBANDS_STEMS = (
    _EGO_SS5_DISTBANDS_TRAIN,
    f"{_EGO_SS5_DISTBANDS_TRAIN}_eval_full",
    "pong_transfer_ego_pov_distbands3_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
    "beamrider_transfer_ego_pov_distbands3_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
    "phoenix_transfer_ego_pov_distbands3_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
)

# Locked clip50 + Machado sticky 0.25 at train (eval sticky=0).
#   python scripts/run_paper_eval.py --suite ego_min_noweapon_surfacesub5_clip50_sticky --gpus 0 --wait-deps --lean
EGO_MIN_NOWEAPON_SURFACESUB5_CLIP50_STICKY_SUITE = "ego_min_noweapon_surfacesub5_clip50_sticky"
_EGO_SS5_CLIP50_STICKY_TRAIN = (
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_surfacesub5_sticky"
)
EGO_MIN_NOWEAPON_SURFACESUB5_CLIP50_STICKY_STEMS = (
    _EGO_SS5_CLIP50_STICKY_TRAIN,
    f"{_EGO_SS5_CLIP50_STICKY_TRAIN}_eval_full",
    "pong_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_surfacesub5_sticky",
    "beamrider_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_surfacesub5_sticky",
    "phoenix_transfer_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_surfacesub5_sticky",
)

# Train-time per-episode clip sampling {10,20,30,50,80}; eval/ZS fixed at clip50.
#   python scripts/materialize_surfacesub5_cliprand.py
#   python scripts/run_paper_eval.py --suite ego_min_noweapon_surfacesub5_cliprand --gpus 0 --wait-deps --lean
EGO_MIN_NOWEAPON_SURFACESUB5_CLIPRAND_SUITE = "ego_min_noweapon_surfacesub5_cliprand"
_EGO_SS5_CLIPRAND_TRAIN = (
    "seaquest_skills_ego_pov_cliprand_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
)
EGO_MIN_NOWEAPON_SURFACESUB5_CLIPRAND_STEMS = (
    _EGO_SS5_CLIPRAND_TRAIN,
    f"{_EGO_SS5_CLIPRAND_TRAIN}_eval_full",
    "pong_transfer_ego_pov_cliprand_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
    "beamrider_transfer_ego_pov_cliprand_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
    "phoenix_transfer_ego_pov_cliprand_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
)

# clip10 surfacesub5 + can_fire only (facing_enemy off). Fun probe: lift SQ/BR?
#   python scripts/materialize_surfacesub5_clip10_canfire.py
#   python scripts/run_paper_eval.py --suite ego_min_surfacesub5_clip10_canfire --gpus 0 --wait-deps --lean
EGO_MIN_SURFACESUB5_CLIP10_CANFIRE_SUITE = "ego_min_surfacesub5_clip10_canfire"
_EGO_SS5_CLIP10_CANFIRE_TRAIN = (
    "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_canfire_surfacesub5"
)
EGO_MIN_SURFACESUB5_CLIP10_CANFIRE_STEMS = (
    _EGO_SS5_CLIP10_CANFIRE_TRAIN,
    f"{_EGO_SS5_CLIP10_CANFIRE_TRAIN}_eval_full",
    "pong_transfer_ego_pov_clip10_facing_psibins8_v2_noblocking_canfire_surfacesub5",
    "beamrider_transfer_ego_pov_clip10_facing_psibins8_v2_noblocking_canfire_surfacesub5",
    "phoenix_transfer_ego_pov_clip10_facing_psibins8_v2_noblocking_canfire_surfacesub5",
)

# Ungated skill-mechanics reward regression (clip10+clip50 lock) + clip10 all-objects.
# Fresh stems (…_skillmech / …_skillmech_relabel); compare to prior clipsweep numbers.
#   python scripts/materialize_surfacesub5_skillmech_rw.py
#   python scripts/run_paper_eval.py --suite ego_min_surfacesub5_skillmech_rw --gpus 0,1,2 --wait-deps --lean
EGO_MIN_SURFACESUB5_SKILLMECH_RW_SUITE = "ego_min_surfacesub5_skillmech_rw"
_EGO_SS5_SKILLMECH_TRAINS = (
    "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_surfacesub5_skillmech",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_surfacesub5_skillmech",
    "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_surfacesub5_skillmech_relabel",
)
EGO_MIN_SURFACESUB5_SKILLMECH_RW_STEMS = (
    _EGO_SS5_SKILLMECH_TRAINS
    + tuple(f"{t}_eval_full" for t in _EGO_SS5_SKILLMECH_TRAINS)
    + tuple(
        f"{env}_transfer_ego_pov_{t.removeprefix('seaquest_skills_ego_pov_')}"
        for t in _EGO_SS5_SKILLMECH_TRAINS
        for env in ("pong", "beamrider", "phoenix")
    )
)

# Retrain + full SQ mods on post-spawn-fix Seaquest (JAXAtari-seaquest-robustness).
# Fresh SAVE_PATHs; clip10 skillmech uses OFF_POLICY=false. Trains first so
# --gpus 2,3 --wait-deps shards one train per GPU, then each eval waits.
#   python scripts/run_paper_eval.py --suite newseaquest_spawnfix \
#     --gpus 2,3 --wait-deps --lean \
#     --jaxatari-root /path/to/JAXAtari-seaquest-robustness
NEWSEAQUEST_SPAWNFIX_SUITE = "newseaquest_spawnfix"
_NEWSEAQUEST_PQN_TRAIN = "seaquest_pqn_newseaquest"
_NEWSEAQUEST_CLIP10_TRAIN = (
    "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_"
    "surfacesub5_skillmech_newseaquest"
)
NEWSEAQUEST_SPAWNFIX_STEMS = (
    _NEWSEAQUEST_PQN_TRAIN,
    _NEWSEAQUEST_CLIP10_TRAIN,
    "seaquest_pqn_eval_newseaquest",
    "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_"
    "surfacesub5_skillmech_eval_full_newseaquest",
)

# ALE CleanRL PQN W&B f163gauq HPs on JAXAtari Seaquest (10M, seed=2).
#   configs/experiment/paper_eval/ale_pqn_match/ARCHITECTURE_COMPARE.md
#   python scripts/run_paper_eval.py --suite ale_pqn_match \
#     --gpus 0 --wait-deps --skip-rq-gate --lean
# Do not pass --seed (YAML SEED=2 matches the ALE run).
ALE_PQN_MATCH_SUITE = "ale_pqn_match"
ALE_PQN_MATCH_STEMS = ("ale_pqn_match_seaquest_seed2",)

# Paper-HP PQN on old_seaquest (dense diver re-arm) × seeds 44/45/46.
# Uses ENV_NAME=old_seaquest in the same robustness jaxatari checkout.
#   python scripts/materialize_seaquest_env_vs_config_seeds.py
#   python scripts/run_paper_eval.py --suite pqn_paperhps_oldseaquest \
#     --gpus 2,3 --wait-deps --lean \
#     --jaxatari-root /path/to/JAXAtari-seaquest-robustness
# Do not pass --seed (YAML SEED per stem).
PQN_PAPERHPS_OLDSEAQUEST_SUITE = "pqn_paperhps_oldseaquest"
_PQN_OLDSEAQUEST_SEEDS = (44, 45, 46)
PQN_PAPERHPS_OLDSEAQUEST_STEMS = tuple(
    f"seaquest_pqn_paperhps_oldseaquest_seed{seed}" for seed in _PQN_OLDSEAQUEST_SEEDS
) + tuple(
    f"seaquest_pqn_eval_paperhps_oldseaquest_seed{seed}"
    for seed in _PQN_OLDSEAQUEST_SEEDS
)

# clip10 skillmech on spawn-fixed Seaquest with OFF_POLICY=true × seeds 44/45/46.
#   python scripts/materialize_seaquest_env_vs_config_seeds.py
#   python scripts/run_paper_eval.py --suite newseaquest_clip10_offpolicy \
#     --gpus 2,3 --wait-deps --lean \
#     --jaxatari-root /path/to/JAXAtari-seaquest-robustness
NEWSEAQUEST_CLIP10_OFFPOLICY_SUITE = "newseaquest_clip10_offpolicy"
_NEWSEAQUEST_CLIP10_OFFPOLICY_SEEDS = (44, 45, 46)
NEWSEAQUEST_CLIP10_OFFPOLICY_STEMS = tuple(
    "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_"
    f"surfacesub5_skillmech_newseaquest_offpolicy_seed{seed}"
    for seed in _NEWSEAQUEST_CLIP10_OFFPOLICY_SEEDS
) + tuple(
    "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_"
    f"surfacesub5_skillmech_eval_full_newseaquest_offpolicy_seed{seed}"
    for seed in _NEWSEAQUEST_CLIP10_OFFPOLICY_SEEDS
)

# Same-scaffolding fixed ego-rule baselines vs clip10 skillmech transfer.
#   python scripts/materialize_fixed_ego_skill_baseline.py
#   python scripts/run_paper_eval.py --suite fixed_ego_skill_baseline --gpus 0 --lean
FIXED_EGO_SKILL_BASELINE_SUITE = "fixed_ego_skill_baseline"
_FIXED_EGO_ARCH = (
    "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_"
    "noblocking_noweapon_surfacesub5_skillmech"
)
FIXED_EGO_SKILL_BASELINE_STEMS = tuple(
    (
        f"{_FIXED_EGO_ARCH}_{mode}_eval_full"
        if env is None
        else (
            f"{env}_transfer_ego_pov_"
            f"{_FIXED_EGO_ARCH.removeprefix('seaquest_skills_ego_pov_')}_{mode}"
        )
    )
    for mode in ("fixed_ego_rules", "constant_ego")
    for env in (None, "pong", "beamrider", "phoenix")
)

# Distance-block pose probes (Δxy / sign(Δxy) / absolute xy) + φ8; clip null.
# Trains first so --gpus 0,1,2 shards one train per GPU.
#   python scripts/materialize_surfacesub5_distpose.py
#   python scripts/run_paper_eval.py --suite ego_min_surfacesub5_distpose --gpus 0,1,2 --wait-deps --lean
EGO_MIN_SURFACESUB5_DISTPOSE_SUITE = "ego_min_surfacesub5_distpose"
_EGO_SS5_DISTPOSE_TOKS = (
    "deltaxy_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
    "deltasign_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
    "posxy_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
)
EGO_MIN_SURFACESUB5_DISTPOSE_STEMS = (
    tuple(f"seaquest_skills_ego_pov_{tok}" for tok in _EGO_SS5_DISTPOSE_TOKS)
    + tuple(
        f"seaquest_skills_ego_pov_{tok}_eval_full" for tok in _EGO_SS5_DISTPOSE_TOKS
    )
    + tuple(
        f"{env}_transfer_ego_pov_{tok}"
        for tok in _EGO_SS5_DISTPOSE_TOKS
        for env in ("pong", "beamrider", "phoenix")
    )
)

# clip10 continuous near + exclusive mid/far tiers (stitch near-field ↔ operational).
#   python scripts/materialize_surfacesub5_neartiers.py
#   python scripts/run_paper_eval.py --suite ego_min_surfacesub5_neartiers --gpus 0 --wait-deps --lean
EGO_MIN_SURFACESUB5_NEARTIERS_SUITE = "ego_min_surfacesub5_neartiers"
_EGO_SS5_NEARTIERS_TRAIN = (
    "seaquest_skills_ego_pov_neartiers_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
)
EGO_MIN_SURFACESUB5_NEARTIERS_STEMS = (
    _EGO_SS5_NEARTIERS_TRAIN,
    f"{_EGO_SS5_NEARTIERS_TRAIN}_eval_full",
    "pong_transfer_ego_pov_neartiers_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
    "beamrider_transfer_ego_pov_neartiers_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
    "phoenix_transfer_ego_pov_neartiers_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
)

# can_fire re-intro overnight: clip10/50 + neartiers + deltaxy ± φ sincos.
#   python scripts/materialize_surfacesub5_canfire_overnight.py
#   python scripts/run_paper_eval.py --suite ego_min_surfacesub5_canfire_overnight \\
#       --gpus 0,1,2 --wait-deps --lean
EGO_MIN_SURFACESUB5_CANFIRE_OVERNIGHT_SUITE = "ego_min_surfacesub5_canfire_overnight"
_EGO_SS5_CANFIRE_OVERNIGHT_TOKS = (
    "clip10_facing_psibins8_v2_noblocking_canfire_surfacesub5",
    "clip50_facing_psibins8_v2_noblocking_canfire_surfacesub5",
    "neartiers_facing_psibins8_v2_noblocking_canfire_surfacesub5",
    "deltaxy_facing_psibins8_v2_noblocking_canfire_surfacesub5",
    "deltaxy_facing_sincos_v2_noblocking_canfire_surfacesub5",
)
EGO_MIN_SURFACESUB5_CANFIRE_OVERNIGHT_STEMS = (
    tuple(f"seaquest_skills_ego_pov_{tok}" for tok in _EGO_SS5_CANFIRE_OVERNIGHT_TOKS)
    + tuple(
        f"seaquest_skills_ego_pov_{tok}_eval_full"
        for tok in _EGO_SS5_CANFIRE_OVERNIGHT_TOKS
    )
    + tuple(
        f"{env}_transfer_ego_pov_{tok}"
        for tok in _EGO_SS5_CANFIRE_OVERNIGHT_TOKS
        for env in ("pong", "beamrider", "phoenix")
    )
)

# ALE clip10 heuristic smoke (+ tandem / SQ-10M / Δxy / Montezuma append).
#   python scripts/materialize_ale_clip10_heu_smoke.py
#   python scripts/run_paper_eval.py --suite ale_clip10_heu_smoke --gpus 0,1,2 --wait-deps --lean

  # Seaquest tandem meta-OC input ablation (oc_simple* + oc_embed control):
  #   python scripts/materialize_seaquest_meta_oc_ablation.py
  #   python scripts/run_paper_eval.py --suite seaquest_meta_oc_ablation \\
  #       --gpus 0,1,2 --wait-deps --lean
# Asteroids/Time Pilot scratch use the same facing spine as other ALE envs.
# SQ ZS source: …_surfacesub5_skillmech (ungated multi-skill rewards).
# Appended: Montezuma scratch; SQ skills 10M (+ sparse C/D/A 10M); Δxy skillmech
# negative control (+ Pong/BR/PH ZS); pixel+ego tandem on SQ+scratch except
# Montezuma (home mod eval + ZS from SQ tandem skills only).
ALE_CLIP10_HEU_SMOKE_SUITE = "ale_clip10_heu_smoke"
_ALE_CLIP10_SPINE = "clip10_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
_ALE_CLIP10_SCRATCH_ENVS = (
    "phoenix",
    "namethisgame",
    "timepilot",
    "asteroids",
    "pong",
    "breakout",
    "beamrider",
    "spaceinvaders",
    "freeway",
    "montezumarevenge",
)
_ALE_CLIP10_TRANSFER_ENVS = (
    "spaceinvaders",
    "beamrider",
    "timepilot",
    "asteroids",
    "pong",
    "breakout",
    "freeway",
    "namethisgame",
    "phoenix",
)
_ALE_CLIP10_NO_TANDEM = frozenset({"montezumarevenge"})
_ALE_CLIP10_TANDEM_ENVS = ("seaquest",) + tuple(
    e for e in _ALE_CLIP10_SCRATCH_ENVS if e not in _ALE_CLIP10_NO_TANDEM
)
_ALE_CLIP10_SQ_10M = (
    "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_"
    "surfacesub5_skillmech_10m"
)
_ALE_CLIP10_SQ_SPARSE_10M = (
    "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_noweapon_"
    "surfacesub5_skillmech_sparse_cda_10m"
)
_ALE_CLIP10_DELTAXY = (
    "seaquest_skills_ego_pov_deltaxy_facing_psibins8_v2_noblocking_noweapon_"
    "surfacesub5_skillmech"
)
_ALE_CLIP10_DELTAXY_ZS = ("pong", "beamrider", "phoenix")


def _ale_clip10_scratch_spine(env: str) -> str:
    del env
    return _ALE_CLIP10_SPINE


def _ale_clip10_tandem_stem(env: str) -> str:
    spine = _ale_clip10_scratch_spine(env) if env != "seaquest" else _ALE_CLIP10_SPINE
    return f"{env}_from_scratch_tandem_pixel_ego_pov_{spine}"


ALE_CLIP10_HEU_SMOKE_STEMS = (
    # --- skills scratch (~25M) ---
    tuple(
        f"{env}_from_scratch_ego_pov_{_ale_clip10_scratch_spine(env)}"
        for env in _ALE_CLIP10_SCRATCH_ENVS
    )
    # --- SQ 10M + sparse 10M + Δxy ---
    + (_ALE_CLIP10_SQ_10M, _ALE_CLIP10_SQ_SPARSE_10M, _ALE_CLIP10_DELTAXY)
    # --- tandem trains (pixel meta + ego) ---
    + tuple(_ale_clip10_tandem_stem(env) for env in _ALE_CLIP10_TANDEM_ENVS)
    # --- ZS from skillmech SQ ---
    + tuple(
        f"{env}_transfer_ego_pov_{_ALE_CLIP10_SPINE}" for env in _ALE_CLIP10_TRANSFER_ENVS
    )
    # --- SQ / sparse / Δxy full mod eval ---
    + (
        f"{_ALE_CLIP10_SQ_10M}_eval_full",
        f"{_ALE_CLIP10_SQ_SPARSE_10M}_eval_full",
        f"{_ALE_CLIP10_DELTAXY}_eval_full",
    )
    # --- Δxy ZS (Pong/BR/PH negative-control transfers) ---
    + tuple(
        f"{env}_transfer_ego_pov_deltaxy_facing_psibins8_v2_"
        f"noblocking_noweapon_surfacesub5_skillmech"
        for env in _ALE_CLIP10_DELTAXY_ZS
    )
    # --- tandem home mod evals ---
    + tuple(f"{_ale_clip10_tandem_stem(env)}_eval" for env in _ALE_CLIP10_TANDEM_ENVS)
    # --- ZS from SQ tandem skills ---
    + tuple(
        f"{env}_transfer_tandem_pixel_ego_pov_{_ALE_CLIP10_SPINE}"
        for env in _ALE_CLIP10_TRANSFER_ENVS
    )
)

# Seaquest tandem meta-OC input ablation (clip10 + sparse C/D/A ~25M + home eval).
# Compare to pixel tandem sparse from sparse_cda_clean.
#   python scripts/materialize_seaquest_meta_oc_ablation.py
#   CUDA_VISIBLE_DEVICES=1 python scripts/run_paper_eval.py \\
#       --suite seaquest_meta_oc_sparse --wait-deps --skip-rq-gate --video
SEAQUEST_META_OC_SPARSE_SUITE = "seaquest_meta_oc_sparse"
# Legacy suite name still works (same sparse stems).
SEAQUEST_META_OC_ABLATION_SUITE = "seaquest_meta_oc_ablation"
_SEAQUEST_META_OC_SPINE = "clip10_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
_SEAQUEST_META_OC_TOKENS = (
    "oc_embed",
    "oc_simple",
    "oc_simple_rel",
    "oc_simple_role",
    "oc_simple_rel_role",
    "oc_simple_inv",
)
SEAQUEST_META_OC_SPARSE_STEMS = tuple(
    f"seaquest_from_scratch_tandem_{tok}_ego_pov_{_SEAQUEST_META_OC_SPINE}_sparse_cda"
    for tok in _SEAQUEST_META_OC_TOKENS
) + tuple(
    f"seaquest_from_scratch_tandem_{tok}_ego_pov_{_SEAQUEST_META_OC_SPINE}_sparse_cda_eval"
    for tok in _SEAQUEST_META_OC_TOKENS
)
SEAQUEST_META_OC_ABLATION_STEMS = SEAQUEST_META_OC_SPARSE_STEMS

# Sparse C/D/A clip50 + clip100 skills (last horizon probe; one GPU sequential).
#   python scripts/materialize_sparse_cda_clip_sweep.py
#   CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \\
#       --suite sparse_cda_clip_sweep --wait-deps --skip-rq-gate --video
SPARSE_CDA_CLIP_SWEEP_SUITE = "sparse_cda_clip_sweep"
SPARSE_CDA_CLIP_TRANSFER_SUITE = "sparse_cda_clip_transfer"
SPARSE_CDA_CLIP_TRANSFER_GPU0_SUITE = "sparse_cda_clip_transfer_gpu0"
SPARSE_CDA_CLIP_TRANSFER_GPU1_SUITE = "sparse_cda_clip_transfer_gpu1"
_SPARSE_CDA_CLIP_SWEEP_CLIPS = (50, 100)
_SPARSE_CDA_CLIP_TRANSFER_ENVS = (
    "spaceinvaders",
    "beamrider",
    "timepilot",
    "asteroids",
    "pong",
    "breakout",
    "freeway",
    "namethisgame",
    "phoenix",
)
SPARSE_CDA_CLIP_SWEEP_STEMS = tuple(
    stem
    for clip in _SPARSE_CDA_CLIP_SWEEP_CLIPS
    for stem in (
        f"seaquest_skills_ego_pov_clip{clip}_facing_psibins8_v2_noblocking_noweapon_"
        f"surfacesub5_skillmech_sparse_cda",
        f"seaquest_skills_ego_pov_clip{clip}_facing_psibins8_v2_noblocking_noweapon_"
        f"surfacesub5_skillmech_sparse_cda_eval_full",
    )
)


def _sparse_cda_clip_transfer_stems(clip: int) -> tuple[str, ...]:
    return tuple(
        f"{env}_transfer_ego_pov_clip{clip}_facing_psibins8_v2_noblocking_noweapon_"
        f"surfacesub5_skillmech_sparse_cda"
        for env in _SPARSE_CDA_CLIP_TRANSFER_ENVS
    )


# Parallel: gpu0 = clip50 ZS (9), gpu1 = clip100 ZS (9). Union kept for one-GPU runs.
SPARSE_CDA_CLIP_TRANSFER_GPU0_STEMS = _sparse_cda_clip_transfer_stems(50)
SPARSE_CDA_CLIP_TRANSFER_GPU1_STEMS = _sparse_cda_clip_transfer_stems(100)
SPARSE_CDA_CLIP_TRANSFER_STEMS = (
    SPARSE_CDA_CLIP_TRANSFER_GPU0_STEMS + SPARSE_CDA_CLIP_TRANSFER_GPU1_STEMS
)

# Sparse C/D/A clip extremes overnight (2 GPUs).
#   python scripts/materialize_sparse_cda_clip_extremes.py
#   CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \
#       --suite sparse_cda_clip_extremes_gpu0 --wait-deps --skip-rq-gate
#   CUDA_VISIBLE_DEVICES=1 python scripts/run_paper_eval.py \
#       --suite sparse_cda_clip_extremes_gpu1 --wait-deps --skip-rq-gate
SPARSE_CDA_CLIP_EXTREMES_SUITE = "sparse_cda_clip_extremes"
SPARSE_CDA_CLIP_EXTREMES_GPU0_SUITE = "sparse_cda_clip_extremes_gpu0"
SPARSE_CDA_CLIP_EXTREMES_GPU1_SUITE = "sparse_cda_clip_extremes_gpu1"
_SPARSE_CDA_EXTREMES_SPINE_CLIP0 = (
    "clip0_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
)
_SPARSE_CDA_EXTREMES_SPINE_NOCLIP = (
    "noclip_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
)
_SPARSE_CDA_EXTREMES_TRANSFER_ENVS = _SPARSE_CDA_CLIP_TRANSFER_ENVS


def _sparse_cda_extremes_clip_pipeline(spine: str) -> tuple[str, ...]:
    train = f"seaquest_skills_ego_pov_{spine}_skillmech_sparse_cda"
    return (train, f"{train}_eval_full") + tuple(
        f"{env}_transfer_ego_pov_{spine}_skillmech_sparse_cda"
        for env in _SPARSE_CDA_EXTREMES_TRANSFER_ENVS
    )


_SPARSE_CDA_TANDEM_FROM_SKILLS = (
    "seaquest_tandem_pixel_from_skills_ego_pov_clip10_facing_psibins8_v2_"
    "noblocking_noweapon_surfacesub5_sparse_cda_20m"
)
# gpu0: clip0 full pipeline. gpu1: noclip full pipeline then tandem-from-skills.
SPARSE_CDA_CLIP_EXTREMES_GPU0_STEMS = _sparse_cda_extremes_clip_pipeline(
    _SPARSE_CDA_EXTREMES_SPINE_CLIP0
)
SPARSE_CDA_CLIP_EXTREMES_GPU1_STEMS = _sparse_cda_extremes_clip_pipeline(
    _SPARSE_CDA_EXTREMES_SPINE_NOCLIP
) + (
    _SPARSE_CDA_TANDEM_FROM_SKILLS,
    f"{_SPARSE_CDA_TANDEM_FROM_SKILLS}_eval",
)
SPARSE_CDA_CLIP_EXTREMES_STEMS = (
    SPARSE_CDA_CLIP_EXTREMES_GPU0_STEMS + SPARSE_CDA_CLIP_EXTREMES_GPU1_STEMS
)

# Noclip + sparse C/D/A full-eval smoke (20M hard; 3 GPUs).
#   python scripts/materialize_noclip_sparse_smoke.py
#   CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \
#       --suite noclip_sparse_smoke_gpu0 --wait-deps --skip-rq-gate
#   CUDA_VISIBLE_DEVICES=1 python scripts/run_paper_eval.py \
#       --suite noclip_sparse_smoke_gpu1 --wait-deps --skip-rq-gate
#   CUDA_VISIBLE_DEVICES=2 python scripts/run_paper_eval.py \
#       --suite noclip_sparse_smoke_gpu2 --wait-deps --skip-rq-gate
# After cascade gaps, fill on one GPU:
#   CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \
#       --suite noclip_sparse_smoke_rerun --wait-deps --skip-rq-gate --resume
# AS/TP tandem catch-up (turn primitives now wired in tandem):
#   CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py \
#       --suite noclip_sparse_smoke_as_tp_scratch --wait-deps --skip-rq-gate
#       --suite noclip_sparse_smoke_as_tp_tandem --wait-deps --skip-rq-gate
NOCLIP_SPARSE_SMOKE_SUITE = "noclip_sparse_smoke"
NOCLIP_SPARSE_SMOKE_GPU0_SUITE = "noclip_sparse_smoke_gpu0"
NOCLIP_SPARSE_SMOKE_GPU1_SUITE = "noclip_sparse_smoke_gpu1"
NOCLIP_SPARSE_SMOKE_GPU2_SUITE = "noclip_sparse_smoke_gpu2"
NOCLIP_SPARSE_SMOKE_RERUN_SUITE = "noclip_sparse_smoke_rerun"
NOCLIP_SPARSE_SMOKE_AS_TP_TANDEM_SUITE = "noclip_sparse_smoke_as_tp_tandem"
NOCLIP_SPARSE_SMOKE_AS_TP_SCRATCH_SUITE = "noclip_sparse_smoke_as_tp_scratch"


def _noclip_sparse_smoke_stems() -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Load stem lists from the materializer (single source of truth)."""
    import sys

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from materialize_noclip_sparse_smoke import (  # type: ignore
        all_stems,
        as_tp_scratch_stems,
        as_tp_tandem_stems,
        gpu0_stems,
        gpu1_stems,
        gpu2_stems,
        rerun_stems,
    )

    return (
        all_stems(),
        gpu0_stems(),
        gpu1_stems(),
        gpu2_stems(),
        rerun_stems(),
        as_tp_tandem_stems(),
        as_tp_scratch_stems(),
    )


(
    NOCLIP_SPARSE_SMOKE_STEMS,
    NOCLIP_SPARSE_SMOKE_GPU0_STEMS,
    NOCLIP_SPARSE_SMOKE_GPU1_STEMS,
    NOCLIP_SPARSE_SMOKE_GPU2_STEMS,
    NOCLIP_SPARSE_SMOKE_RERUN_STEMS,
    NOCLIP_SPARSE_SMOKE_AS_TP_TANDEM_STEMS,
    NOCLIP_SPARSE_SMOKE_AS_TP_SCRATCH_STEMS,
) = _noclip_sparse_smoke_stems()

# Exploration sanity + later HP sweeps (constant SKILL_LR; see materialize_hp_sweep.py).
#   python scripts/materialize_hp_sweep.py
#   python scripts/run_paper_eval.py --suite eps_sanity \\
#       --gpus 0,1,2,3 --wait-deps --skip-rq-gate --lean
EPS_SANITY_SUITE = "eps_sanity"
HP_SWEEP_SKILLS_SUITE = "hp_sweep_skills"
# Clean restart of the mix grid under frozen [32, 64] / death=0 / EGO 18.
HP_SWEEP_TANDEM_SUITE = "hp_sweep_tandem_v2"
# Archival: eval-only recovery for abandoned hp_sweep_tandem [16, 32] ckpts.
HP_SWEEP_TANDEM_CKPT16_SUITE = "hp_sweep_tandem_ckpt16"
# Locked e01 HPs at 20M + ~100M (scale curve after hp_sweep_tandem_v2).
TANDEM_E01_SCALE_SUITE = "tandem_e01_scale"
# Same e01 HPs; three OC metas at 20M (farming / robustness probe).
TANDEM_E01_OC_SCALE_SUITE = "tandem_e01_oc_scale"
# NEXUS-mainline outer loop sanity (tandem first → skills); see materialize_nexus_parity.py.
NEXUS_PARITY_SUITE = "nexus_parity"
# PQN optimal-scaling outer (50M / 1024 envs) × skills+tandem × seeds 44/45/46.
# See materialize_optim_scale.py (1:1 YAML HPs; tandem-first).
OPTIM_SCALE_SQ_SUITE = "optim_scale_sq"
# Skills-only seed-1 HP probes on the same outer (ε / LR ablations).
OPTIM_SCALE_PROBE_SUITE = "optim_scale_probe"
# Post-surface-index fix: 1:1 then εB skills (train→eval interleaved).
OPTIM_SCALE_SKILLS_RERUN_SUITE = "optim_scale_skills_rerun"
# Post-NOOP thesis skills: other bases @ εB; short LR probe; single old YAML-ε.
OPTIM_SCALE_SKILLS_BASES_SUITE = "optim_scale_skills_bases"
OPTIM_SCALE_SKILLS_TRANSFERS_SUITE = "optim_scale_skills_transfers"
OPTIM_SCALE_SKILLS_CLIP_FOILS_SUITE = "optim_scale_skills_clip_foils"
OPTIM_SCALE_SKILLS_LIVING_URGENCY_SUITE = "optim_scale_skills_living_urgency"
OPTIM_SCALE_SKILLS_LR_SUITE = "optim_scale_skills_lr"
OPTIM_SCALE_SKILLS_OLD_EPS_SUITE = "optim_scale_skills_old_eps"
# Flat PQN @ optim_scale 1:1 HPs (50M/1024/32) × 3 seeds + full Seaquest mod eval.
# Run on two GPUs: GPU0=OC, GPU1=pixel (train→eval sequential per seed).
OPTIM_SCALE_PQN_OC_SUITE = "optim_scale_pqn_oc"
OPTIM_SCALE_PQN_PIXEL_SUITE = "optim_scale_pqn_pixel"
# Flat PQN pixel @ 1:1 HPs × 3 seeds on bases+transfers (matched vs skills mods).
OPTIM_SCALE_PQN_PIXEL_ENVS_SUITE = "optim_scale_pqn_pixel_envs"
OPTIM_SCALE_PQN_OC_BUILDS_SUITE = "optim_scale_pqn_oc_builds"
# PQN paper Table 5 outer (128 envs / 32 steps / 100M agent = 400M frames) ×
# flat PQN pixel + matched skills εB × seeds 44/45/46 + home evals.
# See materialize_pqn_paper_table5.py (Seaquest paper ref ≈15899).
PQN_PAPER_T5_SUITE = "pqn_paper_t5"
# PQN pixel trains + evals only (native_downscale=false re-run; skip skills).
PQN_PAPER_T5_PQN_SUITE = "pqn_paper_t5_pqn"
# Heu retrain A/B: seed45 εB skills, noweapon vs can_fire-only (facing off).
# New heuristic (O₂~10%, no-diver→nearest DESTROY) + fixed home mods.
#   python scripts/run_paper_eval.py --suite skills_heu_canfire_ab \
#     --gpus 0 --wait-deps --skip-rq-gate --lean
SKILLS_HEU_CANFIRE_AB_SUITE = "skills_heu_canfire_ab"
SKILLS_HEU_CANFIRE_AB_STEMS = (
    "optim_scale_skills_heu_noweapon_seed45",
    "optim_scale_skills_heu_canfire_seed45",
    "optim_scale_skills_heu_noweapon_seed45_eval",
    "optim_scale_skills_heu_canfire_seed45_eval",
)
# Heu-v2 retrain A/B + home paper_core + ZS transfer × 9 envs (paper_core mods).
#   python scripts/materialize_skills_heu_v2_transfer.py
#   python scripts/run_paper_eval.py --suite skills_heu_v2_transfer_ab \
#     --gpus 0 --wait-deps --skip-rq-gate --lean
SKILLS_HEU_V2_TRANSFER_AB_SUITE = "skills_heu_v2_transfer_ab"


def _skills_heu_v2_transfer_ab_stems() -> tuple[str, ...]:
    import sys

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from materialize_skills_heu_v2_transfer import suite_stems  # type: ignore

    return suite_stems()


SKILLS_HEU_V2_TRANSFER_AB_STEMS = _skills_heu_v2_transfer_ab_stems()
# IB ablation: heu-v2 50M ego vs pixel_embed(+target outline), home paper_core.
#   python scripts/materialize_skills_ib_ego_vs_pixel.py
#   python scripts/run_paper_eval.py --suite skills_ib_ego_vs_pixel \
#     --gpus 0 --wait-deps --skip-rq-gate --lean
SKILLS_IB_EGO_VS_PIXEL_SUITE = "skills_ib_ego_vs_pixel"


def _skills_ib_ego_vs_pixel_stems() -> tuple[str, ...]:
    import sys

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from materialize_skills_ib_ego_vs_pixel import suite_stems  # type: ignore

    return suite_stems()


SKILLS_IB_EGO_VS_PIXEL_STEMS = _skills_ib_ego_vs_pixel_stems()
# Locked 10M ego skills on paper bases only (SQ/PH/NTG/AS); no baselines.
FINAL_10M_BASE_SKILLS_SUITE = "final_10m_base_skills"
# Flat PQN pixel @ 10M on bases (2–2 GPU split) + home evals.
FINAL_10M_PQN_PIXEL_GPU0_SUITE = "final_10m_pqn_pixel_gpu0"
FINAL_10M_PQN_PIXEL_GPU1_SUITE = "final_10m_pqn_pixel_gpu1"
# One-command overnight chains (2 GPUs only).
FINAL_10M_OVERNIGHT_GPU0_SUITE = "final_10m_overnight_gpu0"
FINAL_10M_OVERNIGHT_GPU1_SUITE = "final_10m_overnight_gpu1"


def _hp_sweep_stems() -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    import sys

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from materialize_hp_sweep import (  # type: ignore
        eps_sanity_stems,
        skills_sweep_stems,
        tandem_ckpt16_eval_stems,
        tandem_e01_oc_scale_stems,
        tandem_e01_scale_stems,
        tandem_sweep_stems,
    )

    return (
        eps_sanity_stems(),
        skills_sweep_stems(),
        tandem_sweep_stems(),
        tandem_ckpt16_eval_stems(),
        tandem_e01_scale_stems(),
        tandem_e01_oc_scale_stems(),
    )


(
    EPS_SANITY_STEMS,
    HP_SWEEP_SKILLS_STEMS,
    HP_SWEEP_TANDEM_STEMS,
    HP_SWEEP_TANDEM_CKPT16_STEMS,
    TANDEM_E01_SCALE_STEMS,
    TANDEM_E01_OC_SCALE_STEMS,
) = _hp_sweep_stems()


def _final_eval_suite_stems() -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    import sys

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from materialize_final_10m import (  # type: ignore
        final_10m_base_skills_stems,
        final_10m_pqn_pixel_gpu0_stems,
        final_10m_pqn_pixel_gpu1_stems,
    )
    from materialize_nexus_parity import nexus_parity_stems  # type: ignore
    from materialize_optim_scale import (  # type: ignore
        optim_scale_pqn_oc_builds_stems,
        optim_scale_pqn_oc_stems,
        optim_scale_pqn_pixel_envs_stems,
        optim_scale_pqn_pixel_stems,
        optim_scale_probe_stems,
        optim_scale_skills_bases_stems,
        optim_scale_skills_clip_foils_stems,
        optim_scale_skills_living_urgency_stems,
        optim_scale_skills_lr_stems,
        optim_scale_skills_old_eps_stems,
        optim_scale_skills_rerun_stems,
        optim_scale_skills_transfers_stems,
        optim_scale_sq_stems,
    )

    base = final_10m_base_skills_stems()
    nexus = nexus_parity_stems()
    pqn0 = final_10m_pqn_pixel_gpu0_stems()
    pqn1 = final_10m_pqn_pixel_gpu1_stems()
    optim = optim_scale_sq_stems()
    probe = optim_scale_probe_stems()
    rerun = optim_scale_skills_rerun_stems()
    pqn_oc = optim_scale_pqn_oc_stems()
    pqn_px = optim_scale_pqn_pixel_stems()
    pqn_px_envs = optim_scale_pqn_pixel_envs_stems()
    pqn_oc_builds = optim_scale_pqn_oc_builds_stems()
    bases = optim_scale_skills_bases_stems()
    transfers = optim_scale_skills_transfers_stems()
    clip_foils = optim_scale_skills_clip_foils_stems()
    living = optim_scale_skills_living_urgency_stems()
    lr = optim_scale_skills_lr_stems()
    old_eps = optim_scale_skills_old_eps_stems()
    return (
        base,
        nexus,
        pqn0,
        pqn1,
        nexus + pqn0,
        base + pqn1,
        optim,
        probe,
        rerun,
        pqn_oc,
        pqn_px,
        pqn_px_envs,
        pqn_oc_builds,
        bases,
        transfers,
        clip_foils,
        living,
        lr,
        old_eps,
    )


(
    FINAL_10M_BASE_SKILLS_STEMS,
    NEXUS_PARITY_STEMS,
    FINAL_10M_PQN_PIXEL_GPU0_STEMS,
    FINAL_10M_PQN_PIXEL_GPU1_STEMS,
    FINAL_10M_OVERNIGHT_GPU0_STEMS,
    FINAL_10M_OVERNIGHT_GPU1_STEMS,
    OPTIM_SCALE_SQ_STEMS,
    OPTIM_SCALE_PROBE_STEMS,
    OPTIM_SCALE_SKILLS_RERUN_STEMS,
    OPTIM_SCALE_PQN_OC_STEMS,
    OPTIM_SCALE_PQN_PIXEL_STEMS,
    OPTIM_SCALE_PQN_PIXEL_ENVS_STEMS,
    OPTIM_SCALE_PQN_OC_BUILDS_STEMS,
    OPTIM_SCALE_SKILLS_BASES_STEMS,
    OPTIM_SCALE_SKILLS_TRANSFERS_STEMS,
    OPTIM_SCALE_SKILLS_CLIP_FOILS_STEMS,
    OPTIM_SCALE_SKILLS_LIVING_URGENCY_STEMS,
    OPTIM_SCALE_SKILLS_LR_STEMS,
    OPTIM_SCALE_SKILLS_OLD_EPS_STEMS,
) = _final_eval_suite_stems()


def _pqn_paper_t5_stems() -> tuple[str, ...]:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from materialize_pqn_paper_table5 import pqn_paper_t5_stems  # type: ignore

    return pqn_paper_t5_stems()


def _pqn_paper_t5_pqn_stems() -> tuple[str, ...]:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from materialize_pqn_paper_table5 import pqn_paper_t5_pqn_stems  # type: ignore

    return pqn_paper_t5_pqn_stems()


PQN_PAPER_T5_STEMS = _pqn_paper_t5_stems()
PQN_PAPER_T5_PQN_STEMS = _pqn_paper_t5_pqn_stems()

# Sparse C/D/A clean story (newSeaquest 25M + full eval/ZS; BR+MZ 10M; SQ tandem 20M).
# Uneven pipelines → per-GPU suites (do not use --gpus round-robin on the union).
#   python scripts/materialize_sparse_cda_clean.py
#   bash scripts/run_sparse_cda_clean.sh --gpus 0,1,2
SPARSE_CDA_CLEAN_SUITE = "sparse_cda_clean"
SPARSE_CDA_CLEAN_GPU0_SUITE = "sparse_cda_clean_gpu0"
SPARSE_CDA_CLEAN_GPU1_SUITE = "sparse_cda_clean_gpu1"
SPARSE_CDA_CLEAN_GPU2_SUITE = "sparse_cda_clean_gpu2"
_SPARSE_CDA_SPINE = "clip10_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
_SPARSE_CDA_SQ = (
    f"seaquest_skills_ego_pov_{_SPARSE_CDA_SPINE}_skillmech_sparse_cda"
)
_SPARSE_CDA_TRANSFER_ENVS = (
    "spaceinvaders",
    "beamrider",
    "timepilot",
    "asteroids",
    "pong",
    "breakout",
    "freeway",
    "namethisgame",
    "phoenix",
)
SPARSE_CDA_CLEAN_GPU0_STEMS = (
    _SPARSE_CDA_SQ,
    f"{_SPARSE_CDA_SQ}_eval_full",
) + tuple(
    f"{env}_transfer_ego_pov_{_SPARSE_CDA_SPINE}_skillmech_sparse_cda"
    for env in _SPARSE_CDA_TRANSFER_ENVS
)
SPARSE_CDA_CLEAN_GPU1_STEMS = (
    f"beamrider_from_scratch_ego_pov_{_SPARSE_CDA_SPINE}_sparse_cda_10m",
    f"montezumarevenge_from_scratch_ego_pov_{_SPARSE_CDA_SPINE}_sparse_cda_10m",
)
_SPARSE_CDA_TANDEM = (
    f"seaquest_from_scratch_tandem_pixel_ego_pov_{_SPARSE_CDA_SPINE}_sparse_cda_20m"
)
SPARSE_CDA_CLEAN_GPU2_STEMS = (
    _SPARSE_CDA_TANDEM,
    f"{_SPARSE_CDA_TANDEM}_eval",
)
SPARSE_CDA_CLEAN_STEMS = (
    SPARSE_CDA_CLEAN_GPU0_STEMS
    + SPARSE_CDA_CLEAN_GPU1_STEMS
    + SPARSE_CDA_CLEAN_GPU2_STEMS
)

# Alias: full ground-up batch = tier A (most users start here).
THESIS_FINAL_SUITE = THESIS_TIER_A_SUITE
THESIS_FINAL_STEMS = THESIS_TIER_A_STEMS
DISTBINS8_PIPELINE_SUITE = "distbins8_pipeline"
DISTBINS8_PIPELINE_STEMS = (
    "seaquest_skills_ego_clip50_angbins_distbins8",
    "seaquest_skills_ego_clip50_angbins_distbins8_eval_full",
    "pong_transfer_ego_clip50_angbins_distbins8_full",
    "beamrider_transfer_ego_clip50_angbins_distbins8",
)
# SKILL_TARGET_CUT=lifetime: no meta-switch λ cuts (HER lifetime traces).
KANGAROO_LIFETIME_SUITE = "kangaroo_lifetime"
KANGAROO_LIFETIME_STEMS = (
    "kangaroo_skills_ego_noclip_angbins_lifetime",
)
CLIP50_LIFETIME_PIPELINE_SUITE = "clip50_lifetime_pipeline"
CLIP50_LIFETIME_PIPELINE_STEMS = (
    "seaquest_skills_ego_clip50_angbins_lifetime",
    "seaquest_skills_ego_clip50_angbins_lifetime_eval_full",
    "pong_transfer_ego_clip50_angbins_lifetime_full",
    "phoenix_transfer_ego_clip50_angbins_lifetime",
)
TIERSCALAR_PIPELINE_SUITE = "tierscalar_pipeline"
TIERSCALAR_PIPELINE_STEMS = (
    "seaquest_skills_ego_clip50_angbins_tierscalar",
    "seaquest_skills_ego_clip50_angbins_tierscalar_eval_full",
    "pong_transfer_ego_clip50_angbins_tierscalar_full",
    "beamrider_transfer_ego_clip50_angbins_tierscalar",
)
ANGSCALAR_ONLY_PIPELINE_SUITE = "angscalar_only_pipeline"
ANGSCALAR_ONLY_PIPELINE_STEMS = (
    "seaquest_skills_ego_angscalar_only",
    "seaquest_skills_ego_angscalar_only_eval_full",
    "pong_transfer_ego_angscalar_only_full",
    "beamrider_transfer_ego_angscalar_only",
)
# Two-arm 2-GPU batch: each GPU gets one full arm (train+eval+pong+beam) via sharding.
SCALAR_ABLATION_DUAL_SUITE = "scalar_ablation_dual"
SCALAR_ABLATION_DUAL_STEMS = (
    # idx even -> shard1/GPU0 (tierscalar)
    "seaquest_skills_ego_clip50_angbins_tierscalar",
    # idx odd -> shard2/GPU1 (angscalar-only)
    "seaquest_skills_ego_angscalar_only",
    "seaquest_skills_ego_clip50_angbins_tierscalar_eval_full",
    "seaquest_skills_ego_angscalar_only_eval_full",
    "pong_transfer_ego_clip50_angbins_tierscalar_full",
    "pong_transfer_ego_angscalar_only_full",
    "beamrider_transfer_ego_clip50_angbins_tierscalar",
    "beamrider_transfer_ego_angscalar_only",
)
# Minimal-engineering overnight: 6 isolated ablations off clip50_angbins baseline.
# Stem order is shard-critical for 4 GPUs (i % 4): all trains first (4 parallel),
# then Seaquest eval_full, Pong base, BeamRider base per arm.
MINIMAL_ABLATION_SUITE = "minimal_ablation"
MINIMAL_ABLATION_STEMS = (
    # --- trains (indices 0–5: one on each GPU 0–3, then 4–5 on GPU 0–1) ---
    "seaquest_skills_ego_clip50_ang32scalar",
    "seaquest_skills_ego_clip50_ang32sincos",
    "seaquest_skills_ego_clip50_ang8scalar",
    "seaquest_skills_ego_clip50_angbins_noliving",
    "seaquest_skills_ego_clip50_angbins_nodestroyshape",
    "seaquest_skills_ego_clip50_angbins_nolos",
    # --- Seaquest full mod eval ---
    "seaquest_skills_ego_clip50_ang32scalar_eval_full",
    "seaquest_skills_ego_clip50_ang32sincos_eval_full",
    "seaquest_skills_ego_clip50_ang8scalar_eval_full",
    "seaquest_skills_ego_clip50_angbins_noliving_eval_full",
    "seaquest_skills_ego_clip50_angbins_nodestroyshape_eval_full",
    "seaquest_skills_ego_clip50_angbins_nolos_eval_full",
    # --- Pong base transfer ---
    "pong_transfer_ego_clip50_ang32scalar",
    "pong_transfer_ego_clip50_ang32sincos",
    "pong_transfer_ego_clip50_ang8scalar",
    "pong_transfer_ego_clip50_angbins_noliving",
    "pong_transfer_ego_clip50_angbins_nodestroyshape",
    "pong_transfer_ego_clip50_angbins_nolos",
    # --- BeamRider base transfer ---
    "beamrider_transfer_ego_clip50_ang32scalar",
    "beamrider_transfer_ego_clip50_ang32sincos",
    "beamrider_transfer_ego_clip50_ang8scalar",
    "beamrider_transfer_ego_clip50_angbins_noliving",
    "beamrider_transfer_ego_clip50_angbins_nodestroyshape",
    "beamrider_transfer_ego_clip50_angbins_nolos",
)
# Train-only subset for minimal_ablation (3 GPUs: 2 waves × 3 parallel trains ≈ 3h).
MINIMAL_ABLATION_TRAIN_SUITE = "minimal_ablation_train"
MINIMAL_ABLATION_TRAIN_STEMS = (
    "seaquest_skills_ego_clip50_ang32scalar",
    "seaquest_skills_ego_clip50_ang32sincos",
    "seaquest_skills_ego_clip50_ang8scalar",
    "seaquest_skills_ego_clip50_angbins_noliving",
    "seaquest_skills_ego_clip50_angbins_nodestroyshape",
    "seaquest_skills_ego_clip50_angbins_nolos",
)
# Low distance-clip ablations (clip15 / clip8) for DESTROY range invariance.
# 2 GPUs: trains first, then Seaquest full + Pong/BeamRider/Phoenix base (27k).
CLIP_LOW_PIPELINE_SUITE = "clip_low_pipeline"
CLIP_LOW_PIPELINE_STEMS = (
    # --- trains (2 GPUs) ---
    "seaquest_skills_ego_clip15_angbins",
    "seaquest_skills_ego_clip8_angbins",
    # --- Seaquest full + transfers (legal-ego decode) ---
    "seaquest_skills_ego_clip15_angbins_eval_full",
    "seaquest_skills_ego_clip8_angbins_eval_full",
    "pong_transfer_ego_clip15_angbins",
    "pong_transfer_ego_clip8_angbins",
    "beamrider_transfer_ego_clip15_angbins",
    "beamrider_transfer_ego_clip8_angbins",
    "phoenix_transfer_ego_clip15_angbins",
    "phoenix_transfer_ego_clip8_angbins",
    # --- locked clip50 ckpt re-eval with legal-ego decode (no retrain) ---
    "seaquest_skills_ego_clip50_angbins_legalmask_eval_full",
    "pong_transfer_ego_clip50_angbins_legalmask",
    "beamrider_transfer_ego_clip50_angbins_legalmask",
    "phoenix_transfer_ego_clip50_angbins_legalmask",
)
CLIP_SWEEP_PIPELINE_SUITE = "clip_sweep_pipeline"
CLIP_SWEEP_PIPELINE_STEMS = (
    # --- 7 new trains (2 GPUs) ---
    "seaquest_skills_ego_clip0_angbins",
    "seaquest_skills_ego_clip30_angbins",
    "seaquest_skills_ego_clip60_angbins",
    "seaquest_skills_ego_clip80_angbins",
    "seaquest_skills_ego_clip100_angbins",
    "seaquest_skills_ego_clip120_angbins",
    "seaquest_skills_ego_noclip_angbins",
    # --- existing ckpts: transfer dual-mask (no retrain) ---
    "pong_transfer_ego_clip8_angbins_maskoff",
    "beamrider_transfer_ego_clip8_angbins_maskoff",
    "phoenix_transfer_ego_clip8_angbins_maskoff",
    "pong_transfer_ego_clip8_angbins_maskon",
    "beamrider_transfer_ego_clip8_angbins_maskon",
    "phoenix_transfer_ego_clip8_angbins_maskon",
    "pong_transfer_ego_clip15_angbins_maskoff",
    "beamrider_transfer_ego_clip15_angbins_maskoff",
    "phoenix_transfer_ego_clip15_angbins_maskoff",
    "pong_transfer_ego_clip15_angbins_maskon",
    "beamrider_transfer_ego_clip15_angbins_maskon",
    "phoenix_transfer_ego_clip15_angbins_maskon",
    "pong_transfer_ego_clip50_angbins_maskoff",
    "beamrider_transfer_ego_clip50_angbins_maskoff",
    "phoenix_transfer_ego_clip50_angbins_maskoff",
    "pong_transfer_ego_clip50_angbins_maskon",
    "beamrider_transfer_ego_clip50_angbins_maskon",
    "phoenix_transfer_ego_clip50_angbins_maskon",
    # --- new arms: SQ full (maskoff) + transfer dual-mask ---
    "seaquest_skills_ego_clip0_angbins_eval_full",
    "pong_transfer_ego_clip0_angbins_maskoff",
    "beamrider_transfer_ego_clip0_angbins_maskoff",
    "phoenix_transfer_ego_clip0_angbins_maskoff",
    "pong_transfer_ego_clip0_angbins_maskon",
    "beamrider_transfer_ego_clip0_angbins_maskon",
    "phoenix_transfer_ego_clip0_angbins_maskon",
    "seaquest_skills_ego_clip30_angbins_eval_full",
    "pong_transfer_ego_clip30_angbins_maskoff",
    "beamrider_transfer_ego_clip30_angbins_maskoff",
    "phoenix_transfer_ego_clip30_angbins_maskoff",
    "pong_transfer_ego_clip30_angbins_maskon",
    "beamrider_transfer_ego_clip30_angbins_maskon",
    "phoenix_transfer_ego_clip30_angbins_maskon",
    "seaquest_skills_ego_clip60_angbins_eval_full",
    "pong_transfer_ego_clip60_angbins_maskoff",
    "beamrider_transfer_ego_clip60_angbins_maskoff",
    "phoenix_transfer_ego_clip60_angbins_maskoff",
    "pong_transfer_ego_clip60_angbins_maskon",
    "beamrider_transfer_ego_clip60_angbins_maskon",
    "phoenix_transfer_ego_clip60_angbins_maskon",
    "seaquest_skills_ego_clip80_angbins_eval_full",
    "pong_transfer_ego_clip80_angbins_maskoff",
    "beamrider_transfer_ego_clip80_angbins_maskoff",
    "phoenix_transfer_ego_clip80_angbins_maskoff",
    "pong_transfer_ego_clip80_angbins_maskon",
    "beamrider_transfer_ego_clip80_angbins_maskon",
    "phoenix_transfer_ego_clip80_angbins_maskon",
    "seaquest_skills_ego_clip100_angbins_eval_full",
    "pong_transfer_ego_clip100_angbins_maskoff",
    "beamrider_transfer_ego_clip100_angbins_maskoff",
    "phoenix_transfer_ego_clip100_angbins_maskoff",
    "pong_transfer_ego_clip100_angbins_maskon",
    "beamrider_transfer_ego_clip100_angbins_maskon",
    "phoenix_transfer_ego_clip100_angbins_maskon",
    "seaquest_skills_ego_clip120_angbins_eval_full",
    "pong_transfer_ego_clip120_angbins_maskoff",
    "beamrider_transfer_ego_clip120_angbins_maskoff",
    "phoenix_transfer_ego_clip120_angbins_maskoff",
    "pong_transfer_ego_clip120_angbins_maskon",
    "beamrider_transfer_ego_clip120_angbins_maskon",
    "phoenix_transfer_ego_clip120_angbins_maskon",
    "seaquest_skills_ego_noclip_angbins_eval_full",
    "pong_transfer_ego_noclip_angbins_maskoff",
    "beamrider_transfer_ego_noclip_angbins_maskoff",
    "phoenix_transfer_ego_noclip_angbins_maskoff",
    "pong_transfer_ego_noclip_angbins_maskon",
    "beamrider_transfer_ego_noclip_angbins_maskon",
    "phoenix_transfer_ego_noclip_angbins_maskon",
)

DISTANCE_ENCODE_PIPELINE_SUITE = "distance_encode_pipeline"
DISTANCE_ENCODE_PIPELINE_STEMS = (
    # --- trains (2 GPUs: 5 arms) ---
    "seaquest_skills_ego_angbins_disttier2",
    "seaquest_skills_ego_angbins_disttier4",
    "seaquest_skills_ego_angbins_disttier6",
    "seaquest_skills_ego_angbins_distflags3",
    "seaquest_skills_ego_angbins_distflags6",
    # --- per arm: Seaquest full + Pong/Beam/Phoenix base (maskoff) ---
    "seaquest_skills_ego_angbins_disttier2_eval_full",
    "pong_transfer_ego_angbins_disttier2",
    "beamrider_transfer_ego_angbins_disttier2",
    "phoenix_transfer_ego_angbins_disttier2",
    "seaquest_skills_ego_angbins_disttier4_eval_full",
    "pong_transfer_ego_angbins_disttier4",
    "beamrider_transfer_ego_angbins_disttier4",
    "phoenix_transfer_ego_angbins_disttier4",
    "seaquest_skills_ego_angbins_disttier6_eval_full",
    "pong_transfer_ego_angbins_disttier6",
    "beamrider_transfer_ego_angbins_disttier6",
    "phoenix_transfer_ego_angbins_disttier6",
    "seaquest_skills_ego_angbins_distflags3_eval_full",
    "pong_transfer_ego_angbins_distflags3",
    "beamrider_transfer_ego_angbins_distflags3",
    "phoenix_transfer_ego_angbins_distflags3",
    "seaquest_skills_ego_angbins_distflags6_eval_full",
    "pong_transfer_ego_angbins_distflags6",
    "beamrider_transfer_ego_angbins_distflags6",
    "phoenix_transfer_ego_angbins_distflags6",
)

# POV-relative mapper: train Seaquest 25M + full mod eval + RQ2 transfer (4 envs).
POV_MAPPER_SUITE = "pov_mapper"
POV_MAPPER_STEMS = (
    "seaquest_skills_ego_pov_clip50",
    "seaquest_skills_ego_pov_clip50_eval_full",
    "pong_transfer_ego_pov_clip50",
    "beamrider_transfer_ego_pov_clip50",
    "asterix_transfer_ego_pov_clip50",
    "freeway_transfer_ego_pov_clip50",
)
# POV-relative mapper, unclipped distance: train + full mod eval only.
POV_NOCLIP_SUITE = "pov_noclip"
POV_NOCLIP_STEMS = (
    "seaquest_skills_ego_pov_noclip",
    "seaquest_skills_ego_pov_noclip_eval_full",
    "pong_transfer_ego_pov_noclip",
)
POV_NOCLIP_EVAL_SUITE = "pov_noclip_eval"
POV_NOCLIP_EVAL_STEMS = ("seaquest_skills_ego_pov_noclip_eval_full",)

# POV baseline recovery: 3 parallel trains, then full Seaquest mod eval + Pong zs per
# arm, plus the angbins reference eval.
#
# The arms form a ladder on top of POV clip50:
#   psibins8            — ψ-bins instead of sin/cos(ψ) (angle encoding only)
#   vel_psibins8        — + signed view-radial approach velocity
#   vel_psibins8_turn   — same obs as vel_psibins8, but 6 turn primitives over a
#                         persistent heading instead of 18 compass intentions.
#                         Paired with vel_psibins8 this isolates the action space.
#
# Dropped from the earlier plan: the noclip train (clip50 is the better-understood
# baseline) and the vel-only train (subsumed by vel_psibins8).
POV_BASELINE_SUITE = "pov_baseline"
POV_BASELINE_STEMS = (
    # --- trains (indices 0–2 → GPUs 0–2) ---
    "seaquest_skills_ego_pov_clip50_psibins8",
    "seaquest_skills_ego_pov_clip50_vel_psibins8",
    "seaquest_skills_ego_pov_clip50_vel_psibins8_turn",
    # --- Seaquest full mod eval @ 27k ---
    "seaquest_skills_ego_pov_clip50_psibins8_eval_full",
    "seaquest_skills_ego_pov_clip50_vel_psibins8_eval_full",
    "seaquest_skills_ego_pov_clip50_vel_psibins8_turn_eval_full",
    # --- Pong zero-shot (transfer sanity) ---
    "pong_transfer_ego_pov_clip50_psibins8",
    "pong_transfer_ego_pov_clip50_vel_psibins8",
    "pong_transfer_ego_pov_clip50_vel_psibins8_turn",
    # --- Reference baseline (existing ckpts, eval-only) ---
    "seaquest_skills_ego_clip50_angbins_eval_full",
    "pong_transfer_ego_clip50_angbins",
)
POV_BASELINE_TRAIN_SUITE = "pov_baseline_train"
POV_BASELINE_TRAIN_STEMS = (
    "seaquest_skills_ego_pov_clip50_psibins8",
    "seaquest_skills_ego_pov_clip50_vel_psibins8",
    "seaquest_skills_ego_pov_clip50_vel_psibins8_turn",
)
POV_VEL_SUITE = "pov_vel"
POV_VEL_STEMS = (
    "seaquest_skills_ego_pov_clip50_vel",
    "seaquest_skills_ego_pov_clip50_vel_eval_full",
    "pong_transfer_ego_pov_clip50_vel",
)
POV_PSIBINS8_SUITE = "pov_psibins8"
POV_PSIBINS8_STEMS = (
    "seaquest_skills_ego_pov_clip50_psibins8",
    "seaquest_skills_ego_pov_clip50_psibins8_eval_full",
    "pong_transfer_ego_pov_clip50_psibins8",
)
POV_VEL_PSIBINS8_SUITE = "pov_vel_psibins8"
POV_VEL_PSIBINS8_STEMS = (
    "seaquest_skills_ego_pov_clip50_vel_psibins8",
    "seaquest_skills_ego_pov_clip50_vel_psibins8_eval_full",
    "pong_transfer_ego_pov_clip50_vel_psibins8",
)
# Virtual bilateral sectors (target-selected L/R or U/D) + forward-hemisphere φ.
# Order is shard-critical: 3 GPUs each own one full arm (train → eval → pong → beam).
POV_FIXED2_SUITE = "pov_fixed2"
POV_FIXED2_STEMS = (
    # --- trains (indices 0–2 → GPUs 0–2) ---
    "seaquest_skills_ego_pov_clip50_fixed2_psibins8",
    "seaquest_skills_ego_pov_clip50_fixed2_sincos",
    "seaquest_skills_ego_pov_clip50_fixed2_vert_psibins8",
    # --- Seaquest full mod eval @ 27k ---
    "seaquest_skills_ego_pov_clip50_fixed2_psibins8_eval_full",
    "seaquest_skills_ego_pov_clip50_fixed2_sincos_eval_full",
    "seaquest_skills_ego_pov_clip50_fixed2_vert_psibins8_eval_full",
    # --- Pong zero-shot ---
    "pong_transfer_ego_pov_clip50_fixed2_psibins8",
    "pong_transfer_ego_pov_clip50_fixed2_sincos",
    "pong_transfer_ego_pov_clip50_fixed2_vert_psibins8",
    # --- Beamrider zero-shot ---
    "beamrider_transfer_ego_pov_clip50_fixed2_psibins8",
    "beamrider_transfer_ego_pov_clip50_fixed2_sincos",
    "beamrider_transfer_ego_pov_clip50_fixed2_vert_psibins8",
)
POV_FIXED2_TRAIN_SUITE = "pov_fixed2_train"
POV_FIXED2_TRAIN_STEMS = (
    "seaquest_skills_ego_pov_clip50_fixed2_psibins8",
    "seaquest_skills_ego_pov_clip50_fixed2_sincos",
    "seaquest_skills_ego_pov_clip50_fixed2_vert_psibins8",
)

# Facing-locked FPS ego (view=player facing). Two pose encodings:
#   facing_psibins8 — distance + 8 full-circle φ bins
#   facing_xy       — body-frame (fwd, lat), no angle
# Order is shard-critical: trains first so 2 GPUs train in parallel.
POV_FACING_SUITE = "pov_facing"
POV_FACING_STEMS = (
    # --- trains (indices 0–1 → GPUs 0–1) ---
    "seaquest_skills_ego_pov_clip50_facing_psibins8",
    "seaquest_skills_ego_pov_clip50_facing_xy",
    # --- Seaquest full mod eval @ 27k ---
    "seaquest_skills_ego_pov_clip50_facing_psibins8_eval_full",
    "seaquest_skills_ego_pov_clip50_facing_xy_eval_full",
    # --- Pong zero-shot ---
    "pong_transfer_ego_pov_clip50_facing_psibins8",
    "pong_transfer_ego_pov_clip50_facing_xy",
    # --- Beamrider zero-shot ---
    "beamrider_transfer_ego_pov_clip50_facing_psibins8",
    "beamrider_transfer_ego_pov_clip50_facing_xy",
)
POV_FACING_TRAIN_SUITE = "pov_facing_train"
POV_FACING_TRAIN_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_psibins8",
    "seaquest_skills_ego_pov_clip50_facing_xy",
)

# Facing FPS + soft midfield COLLECT curriculum (edge-margin heuristic).
# Arms: xy+mid, φ-bins8+mid, xy+clip100 (no mid), xy+noclip (no mid).
# Order is shard-critical: trains first so GPUs start on training.
POV_FACING_COLLECT_SUITE = "pov_facing_collect"
POV_FACING_COLLECT_STEMS = (
    # --- trains (indices 0–3; with 3 GPUs the 4th waits / resumes) ---
    "seaquest_skills_ego_pov_clip50_facing_xy_mid",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_mid",
    "seaquest_skills_ego_pov_clip100_facing_xy",
    "seaquest_skills_ego_pov_noclip_facing_xy",
    # --- Seaquest full mod eval @ 27k ---
    "seaquest_skills_ego_pov_clip50_facing_xy_mid_eval_full",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_mid_eval_full",
    "seaquest_skills_ego_pov_clip100_facing_xy_eval_full",
    "seaquest_skills_ego_pov_noclip_facing_xy_eval_full",
    # --- Pong zero-shot ---
    "pong_transfer_ego_pov_clip50_facing_xy_mid",
    "pong_transfer_ego_pov_clip50_facing_psibins8_mid",
    "pong_transfer_ego_pov_clip100_facing_xy",
    "pong_transfer_ego_pov_noclip_facing_xy",
    # --- Beamrider zero-shot ---
    "beamrider_transfer_ego_pov_clip50_facing_xy_mid",
    "beamrider_transfer_ego_pov_clip50_facing_psibins8_mid",
    "beamrider_transfer_ego_pov_clip100_facing_xy",
    "beamrider_transfer_ego_pov_noclip_facing_xy",
)
POV_FACING_COLLECT_TRAIN_SUITE = "pov_facing_collect_train"
POV_FACING_COLLECT_TRAIN_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_xy_mid",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_mid",
    "seaquest_skills_ego_pov_clip100_facing_xy",
    "seaquest_skills_ego_pov_noclip_facing_xy",
)

# Post-facingfix retrain (obs/mapper aligned). See docs/POV_FRAME_BACKLOG.md.
# Phase 1: φ-bins8 + mid curriculum — train Seaquest, eval mods, Pong zero-shot.
# Phase 2: add facing_xy + Beamrider transfers.
POV_FACING_V2_SUITE = "pov_facing_v2"
POV_FACING_V2_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_mid_v2",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_eval_full",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_mid_v2_eval_full",
    "pong_transfer_ego_pov_clip50_facing_psibins8_v2",
    "pong_transfer_ego_pov_clip50_facing_psibins8_mid_v2",
)
POV_FACING_V2_TRAIN_SUITE = "pov_facing_v2_train"
POV_FACING_V2_TRAIN_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_mid_v2",
)
# Re-run Seaquest mod eval after ckpt-path fix (no retraining).
POV_FACING_V2_EVAL_SUITE = "pov_facing_v2_eval"
POV_FACING_V2_EVAL_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_eval_full",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_mid_v2_eval_full",
)
POV_FACING_V2_PHASE2_SUITE = "pov_facing_v2_phase2"
POV_FACING_V2_PHASE2_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_xy_v2",
    "seaquest_skills_ego_pov_clip50_facing_xy_v2_eval_full",
    "pong_transfer_ego_pov_clip50_facing_xy_v2",
    "beamrider_transfer_ego_pov_clip50_facing_psibins8_v2",
    "beamrider_transfer_ego_pov_clip50_facing_xy_v2",
)

# Three pinned tmux sessions (do NOT use --gpus; keep chains on one device each):
#   CUDA_VISIBLE_DEVICES=0 … --suite pov_facing_v2_gpu0
#   CUDA_VISIBLE_DEVICES=1 … --suite pov_facing_v2_gpu1
#   CUDA_VISIBLE_DEVICES=2 … --suite pov_facing_v2_gpu2
# GPU0: φ-bins8 zero-shot BR+PH (base) → Phoenix from-scratch facing φ-bins8
POV_FACING_V2_GPU0_SUITE = "pov_facing_v2_gpu0"
POV_FACING_V2_GPU0_STEMS = (
    "beamrider_transfer_ego_pov_clip50_facing_psibins8_v2",
    "phoenix_transfer_ego_pov_clip50_facing_psibins8_v2",
    "phoenix_from_scratch_ego_pov_facing_psibins8",
)
# GPU1: φ-bins8+mid zero-shot BR+PH (base) → Beamrider from-scratch facing φ-bins8
POV_FACING_V2_GPU1_SUITE = "pov_facing_v2_gpu1"
POV_FACING_V2_GPU1_STEMS = (
    "beamrider_transfer_ego_pov_clip50_facing_psibins8_mid_v2",
    "phoenix_transfer_ego_pov_clip50_facing_psibins8_mid_v2",
    "beamrider_from_scratch_ego_pov_facing_psibins8",
)
# GPU2: Seaquest noclip facing_xy train → SQ mods + Pong/PH/BR zero-shot base
POV_FACING_V2_GPU2_SUITE = "pov_facing_v2_gpu2"
POV_FACING_V2_GPU2_STEMS = (
    "seaquest_skills_ego_pov_noclip_facing_xy_v2",
    "seaquest_skills_ego_pov_noclip_facing_xy_v2_eval_full",
    "pong_transfer_ego_pov_noclip_facing_xy_v2",
    "phoenix_transfer_ego_pov_noclip_facing_xy_v2",
    "beamrider_transfer_ego_pov_noclip_facing_xy_v2",
)

# Facing ablations: clip20 / noclip / sincos / φ-deg-scalar (each full pipeline).
POV_FACING_ABLATION_SUITE = "pov_facing_ablation"
POV_FACING_ABLATION_STEMS = (
    "seaquest_skills_ego_pov_clip20_facing_psibins8_v2",
    "seaquest_skills_ego_pov_clip20_facing_psibins8_v2_eval_full",
    "pong_transfer_ego_pov_clip20_facing_psibins8_v2",
    "phoenix_transfer_ego_pov_clip20_facing_psibins8_v2",
    "seaquest_skills_ego_pov_noclip_facing_psibins8_v2",
    "seaquest_skills_ego_pov_noclip_facing_psibins8_v2_eval_full",
    "pong_transfer_ego_pov_noclip_facing_psibins8_v2",
    "phoenix_transfer_ego_pov_noclip_facing_psibins8_v2",
    "seaquest_skills_ego_pov_clip50_facing_sincos_v2",
    "seaquest_skills_ego_pov_clip50_facing_sincos_v2_eval_full",
    "pong_transfer_ego_pov_clip50_facing_sincos_v2",
    "phoenix_transfer_ego_pov_clip50_facing_sincos_v2",
    "seaquest_skills_ego_pov_clip50_facing_phideg_v2",
    "seaquest_skills_ego_pov_clip50_facing_phideg_v2_eval_full",
    "pong_transfer_ego_pov_clip50_facing_phideg_v2",
    "phoenix_transfer_ego_pov_clip50_facing_phideg_v2",
)
# Three pinned GPUs + phase2 for φ-deg (or run full suite serially):
#   CUDA_VISIBLE_DEVICES=0 … --suite pov_facing_ablation_gpu0
#   CUDA_VISIBLE_DEVICES=1 … --suite pov_facing_ablation_gpu1
#   CUDA_VISIBLE_DEVICES=2 … --suite pov_facing_ablation_gpu2
#   CUDA_VISIBLE_DEVICES=0 … --suite pov_facing_ablation_phideg  (after gpu0 train)
POV_FACING_ABLATION_GPU0_SUITE = "pov_facing_ablation_gpu0"
POV_FACING_ABLATION_GPU0_STEMS = (
    "seaquest_skills_ego_pov_clip20_facing_psibins8_v2",
    "seaquest_skills_ego_pov_clip20_facing_psibins8_v2_eval_full",
    "pong_transfer_ego_pov_clip20_facing_psibins8_v2",
    "phoenix_transfer_ego_pov_clip20_facing_psibins8_v2",
)
POV_FACING_ABLATION_GPU1_SUITE = "pov_facing_ablation_gpu1"
POV_FACING_ABLATION_GPU1_STEMS = (
    "seaquest_skills_ego_pov_noclip_facing_psibins8_v2",
    "seaquest_skills_ego_pov_noclip_facing_psibins8_v2_eval_full",
    "pong_transfer_ego_pov_noclip_facing_psibins8_v2",
    "phoenix_transfer_ego_pov_noclip_facing_psibins8_v2",
)
POV_FACING_ABLATION_GPU2_SUITE = "pov_facing_ablation_gpu2"
POV_FACING_ABLATION_GPU2_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_sincos_v2",
    "seaquest_skills_ego_pov_clip50_facing_sincos_v2_eval_full",
    "pong_transfer_ego_pov_clip50_facing_sincos_v2",
    "phoenix_transfer_ego_pov_clip50_facing_sincos_v2",
)
POV_FACING_ABLATION_PHIDEG_SUITE = "pov_facing_ablation_phideg"
POV_FACING_ABLATION_PHIDEG_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_phideg_v2",
    "seaquest_skills_ego_pov_clip50_facing_phideg_v2_eval_full",
    "pong_transfer_ego_pov_clip50_facing_phideg_v2",
    "phoenix_transfer_ego_pov_clip50_facing_phideg_v2",
)

# Honest-transfer reward ablations on facing φ8 clip50 v2 spine (sequential):
#   Arm1 — COLLECT_SPARSE_ONLY (keep COLLECT death + DESTROY living/contact)
#   Arm2 — Arm1 + DESTROY sparse (no living tax / contact-fail)
# See tmp/collect_sparse_arms_LOG.md. Materialize:
#   python scripts/materialize_collect_sparse_arms.py
COLLECT_SPARSE_ARMS_SUITE = "collect_sparse_arms"
COLLECT_SPARSE_ARMS_STEMS = (
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_collect_sparse",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_collect_sparse_eval_full",
    "pong_transfer_ego_pov_clip50_facing_psibins8_v2_collect_sparse",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_collect_sparse_destroy_sparse",
    "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_collect_sparse_destroy_sparse_eval_full",
    "pong_transfer_ego_pov_clip50_facing_psibins8_v2_collect_sparse_destroy_sparse",
)

# Arm3: restrained L/C/R × near/far pose on facing v2 spine + legal ego mask.
# Materialize: python scripts/materialize_restrained_ego_suite.py
RESTRAINED_EGO_SUITE = "restrained_ego"
RESTRAINED_EGO_STEMS = (
    "seaquest_skills_ego_pov_facing_lcr_nearfar_v2",
    "seaquest_skills_ego_pov_facing_lcr_nearfar_v2_eval_full",
    "pong_transfer_ego_pov_facing_lcr_nearfar_v2",
)

# Eval-only ablation: reuse seaquest_skills_pixel ckpt with target outline OFF
# (SKILL_PIXEL_PAINT_TARGET_OUTLINE: false) on Seaquest mods + RQ2 zero-shot.
PIXEL_NOOUTLINE_SUITE = "pixel_nooutline"
PIXEL_NOOUTLINE_STEMS = (
    "seaquest_skills_pixel_eval_nooutline",
    "pong_transfer_pixel_nooutline",
    "tennis_transfer_pixel_nooutline",
    "beamrider_transfer_pixel_nooutline",
    "freeway_transfer_pixel_nooutline",
    "asterix_transfer_pixel_nooutline",
    "phoenix_transfer_pixel_nooutline",
)

# Seaquest flat PQN zero-shot on RQ2 envs (control: transfer should be near floor).
PQN_TRANSFER_SUITE = "pqn_transfer"
PQN_TRANSFER_STEMS = (
    "pong_transfer_pqn",
    "tennis_transfer_pqn",
    "beamrider_transfer_pqn",
    "freeway_transfer_pqn",
    "asterix_transfer_pqn",
    "phoenix_transfer_pqn",
)

SUITE_STEMS = {
    SKILL_RESHAPE_SUITE: SKILL_RESHAPE_STEMS,
    DESTROY_ABLATION_SUITE: DESTROY_ABLATION_STEMS,
    DESTROY_URGENCY_SUITE: DESTROY_URGENCY_STEMS,
    COLLECT_SHOT_PROBES_SUITE: COLLECT_SHOT_PROBES_STEMS,
    EGO_POSE_PROBES_SUITE: EGO_POSE_PROBES_STEMS,
    COLLECT_COMBAT_PROBES_SUITE: COLLECT_COMBAT_PROBES_STEMS,
    NO_SURFACE_DEATH_SUITE: NO_SURFACE_DEATH_STEMS,
    EGO_CLIP_BINS_SUITE: EGO_CLIP_BINS_STEMS,
    CLIP_BINS_NODECAY_SUITE: CLIP_BINS_NODECAY_STEMS,
    CLIP50_ANGBINS_FULL_SUITE: CLIP50_ANGBINS_FULL_STEMS,
    THESIS_TIER_A_SUITE: THESIS_TIER_A_STEMS,
    THESIS_FACING_SUITE: THESIS_FACING_STEMS,
    THESIS_TIER_B_SUITE: THESIS_TIER_B_STEMS,
    THESIS_TIER_B_FACING_SUITE: THESIS_TIER_B_FACING_STEMS,
    CLIP100_COLLECTSURFACE_PROBE_SUITE: CLIP100_COLLECTSURFACE_PROBE_STEMS,
    CLIP150_COLLECTSURFACE_PROBE_SUITE: CLIP150_COLLECTSURFACE_PROBE_STEMS,
    FACING_DISTANCE_ABLATION_SUITE: FACING_DISTANCE_ABLATION_STEMS,
    THESIS_TIER_C_SUITE: THESIS_TIER_C_STEMS,
    THESIS_TIER_C_FACING_SUITE: THESIS_TIER_C_FACING_STEMS,
    NAVIGATE_SURFACE_TANDEM_PIXEL_SUITE: NAVIGATE_SURFACE_TANDEM_PIXEL_STEMS,
    NAVIGATE_SURFACE_SURFACEGAP_SUITE: NAVIGATE_SURFACE_SURFACEGAP_STEMS,
    SURFACEGAP_GEOM_DUAL_SUITE: SURFACEGAP_GEOM_DUAL_STEMS,
    NAVIGATE_SURFACE_CENTRES_SUITE: NAVIGATE_SURFACE_CENTRES_STEMS,
    NAVIGATE_SURFACE_BODYGAPS_SUITE: NAVIGATE_SURFACE_BODYGAPS_STEMS,
    EGO_MIN_CHANNELS_SUITE: EGO_MIN_CHANNELS_STEMS,
    EGO_MIN_DESTROYEXTRAS_SUITE: EGO_MIN_DESTROYEXTRAS_STEMS,
    EGO_MIN_NOCLIP_SUITE: EGO_MIN_NOCLIP_STEMS,
    EGO_MIN_NOWEAPON_OVALAVOID_SUITE: EGO_MIN_NOWEAPON_OVALAVOID_STEMS,
    EGO_MIN_NOWEAPON_OVALAVOID_SEEDS_SUITE: EGO_MIN_NOWEAPON_OVALAVOID_SEEDS_STEMS,
    EGO_MIN_NOWEAPON_HEU_ABLATION_SUITE: EGO_MIN_NOWEAPON_HEU_ABLATION_STEMS,
    EGO_MIN_NOFACING_OVALAVOID_SUITE: EGO_MIN_NOFACING_OVALAVOID_STEMS,
    EGO_MIN_NOWEAPON_SURFACESUB5_CLIPSWEEP_SUITE: EGO_MIN_NOWEAPON_SURFACESUB5_CLIPSWEEP_STEMS,
    # deprecated alias → same stems
    "ego_min_noweapon_ovalavoid_clipsweep": EGO_MIN_NOWEAPON_SURFACESUB5_CLIPSWEEP_STEMS,
    EGO_MIN_NOWEAPON_SURFACESUB5_DISTABL_SUITE: EGO_MIN_NOWEAPON_SURFACESUB5_DISTABL_STEMS,
    EGO_MIN_NOWEAPON_SURFACESUB5_DISTBANDS_SUITE: EGO_MIN_NOWEAPON_SURFACESUB5_DISTBANDS_STEMS,
    EGO_MIN_NOWEAPON_SURFACESUB5_CLIP50_STICKY_SUITE: EGO_MIN_NOWEAPON_SURFACESUB5_CLIP50_STICKY_STEMS,
    EGO_MIN_NOWEAPON_SURFACESUB5_CLIPRAND_SUITE: EGO_MIN_NOWEAPON_SURFACESUB5_CLIPRAND_STEMS,
    EGO_MIN_SURFACESUB5_CLIP10_CANFIRE_SUITE: EGO_MIN_SURFACESUB5_CLIP10_CANFIRE_STEMS,
    EGO_MIN_SURFACESUB5_SKILLMECH_RW_SUITE: EGO_MIN_SURFACESUB5_SKILLMECH_RW_STEMS,
    NEWSEAQUEST_SPAWNFIX_SUITE: NEWSEAQUEST_SPAWNFIX_STEMS,
    ALE_PQN_MATCH_SUITE: ALE_PQN_MATCH_STEMS,
    PQN_PAPERHPS_OLDSEAQUEST_SUITE: PQN_PAPERHPS_OLDSEAQUEST_STEMS,
    NEWSEAQUEST_CLIP10_OFFPOLICY_SUITE: NEWSEAQUEST_CLIP10_OFFPOLICY_STEMS,
    FIXED_EGO_SKILL_BASELINE_SUITE: FIXED_EGO_SKILL_BASELINE_STEMS,
    EGO_MIN_SURFACESUB5_DISTPOSE_SUITE: EGO_MIN_SURFACESUB5_DISTPOSE_STEMS,
    EGO_MIN_SURFACESUB5_NEARTIERS_SUITE: EGO_MIN_SURFACESUB5_NEARTIERS_STEMS,
    EGO_MIN_SURFACESUB5_CANFIRE_OVERNIGHT_SUITE: EGO_MIN_SURFACESUB5_CANFIRE_OVERNIGHT_STEMS,
    ALE_CLIP10_HEU_SMOKE_SUITE: ALE_CLIP10_HEU_SMOKE_STEMS,
    SEAQUEST_META_OC_ABLATION_SUITE: SEAQUEST_META_OC_ABLATION_STEMS,
    SEAQUEST_META_OC_SPARSE_SUITE: SEAQUEST_META_OC_SPARSE_STEMS,
    SPARSE_CDA_CLIP_SWEEP_SUITE: SPARSE_CDA_CLIP_SWEEP_STEMS,
    SPARSE_CDA_CLIP_TRANSFER_SUITE: SPARSE_CDA_CLIP_TRANSFER_STEMS,
    SPARSE_CDA_CLIP_TRANSFER_GPU0_SUITE: SPARSE_CDA_CLIP_TRANSFER_GPU0_STEMS,
    SPARSE_CDA_CLIP_TRANSFER_GPU1_SUITE: SPARSE_CDA_CLIP_TRANSFER_GPU1_STEMS,
    SPARSE_CDA_CLIP_EXTREMES_SUITE: SPARSE_CDA_CLIP_EXTREMES_STEMS,
    SPARSE_CDA_CLIP_EXTREMES_GPU0_SUITE: SPARSE_CDA_CLIP_EXTREMES_GPU0_STEMS,
    SPARSE_CDA_CLIP_EXTREMES_GPU1_SUITE: SPARSE_CDA_CLIP_EXTREMES_GPU1_STEMS,
    NOCLIP_SPARSE_SMOKE_SUITE: NOCLIP_SPARSE_SMOKE_STEMS,
    NOCLIP_SPARSE_SMOKE_GPU0_SUITE: NOCLIP_SPARSE_SMOKE_GPU0_STEMS,
    NOCLIP_SPARSE_SMOKE_GPU1_SUITE: NOCLIP_SPARSE_SMOKE_GPU1_STEMS,
    NOCLIP_SPARSE_SMOKE_GPU2_SUITE: NOCLIP_SPARSE_SMOKE_GPU2_STEMS,
    NOCLIP_SPARSE_SMOKE_RERUN_SUITE: NOCLIP_SPARSE_SMOKE_RERUN_STEMS,
    NOCLIP_SPARSE_SMOKE_AS_TP_TANDEM_SUITE: NOCLIP_SPARSE_SMOKE_AS_TP_TANDEM_STEMS,
    NOCLIP_SPARSE_SMOKE_AS_TP_SCRATCH_SUITE: NOCLIP_SPARSE_SMOKE_AS_TP_SCRATCH_STEMS,
    EPS_SANITY_SUITE: EPS_SANITY_STEMS,
    HP_SWEEP_SKILLS_SUITE: HP_SWEEP_SKILLS_STEMS,
    HP_SWEEP_TANDEM_SUITE: HP_SWEEP_TANDEM_STEMS,
    HP_SWEEP_TANDEM_CKPT16_SUITE: HP_SWEEP_TANDEM_CKPT16_STEMS,
    TANDEM_E01_SCALE_SUITE: TANDEM_E01_SCALE_STEMS,
    TANDEM_E01_OC_SCALE_SUITE: TANDEM_E01_OC_SCALE_STEMS,
    NEXUS_PARITY_SUITE: NEXUS_PARITY_STEMS,
    OPTIM_SCALE_SQ_SUITE: OPTIM_SCALE_SQ_STEMS,
    OPTIM_SCALE_PROBE_SUITE: OPTIM_SCALE_PROBE_STEMS,
    OPTIM_SCALE_SKILLS_RERUN_SUITE: OPTIM_SCALE_SKILLS_RERUN_STEMS,
    OPTIM_SCALE_SKILLS_BASES_SUITE: OPTIM_SCALE_SKILLS_BASES_STEMS,
    OPTIM_SCALE_SKILLS_TRANSFERS_SUITE: OPTIM_SCALE_SKILLS_TRANSFERS_STEMS,
    OPTIM_SCALE_SKILLS_CLIP_FOILS_SUITE: OPTIM_SCALE_SKILLS_CLIP_FOILS_STEMS,
    OPTIM_SCALE_SKILLS_LIVING_URGENCY_SUITE: OPTIM_SCALE_SKILLS_LIVING_URGENCY_STEMS,
    OPTIM_SCALE_SKILLS_LR_SUITE: OPTIM_SCALE_SKILLS_LR_STEMS,
    OPTIM_SCALE_SKILLS_OLD_EPS_SUITE: OPTIM_SCALE_SKILLS_OLD_EPS_STEMS,
    OPTIM_SCALE_PQN_OC_SUITE: OPTIM_SCALE_PQN_OC_STEMS,
    OPTIM_SCALE_PQN_PIXEL_SUITE: OPTIM_SCALE_PQN_PIXEL_STEMS,
    OPTIM_SCALE_PQN_PIXEL_ENVS_SUITE: OPTIM_SCALE_PQN_PIXEL_ENVS_STEMS,
    OPTIM_SCALE_PQN_OC_BUILDS_SUITE: OPTIM_SCALE_PQN_OC_BUILDS_STEMS,
    PQN_PAPER_T5_SUITE: PQN_PAPER_T5_STEMS,
    PQN_PAPER_T5_PQN_SUITE: PQN_PAPER_T5_PQN_STEMS,
    SKILLS_HEU_CANFIRE_AB_SUITE: SKILLS_HEU_CANFIRE_AB_STEMS,
    SKILLS_HEU_V2_TRANSFER_AB_SUITE: SKILLS_HEU_V2_TRANSFER_AB_STEMS,
    SKILLS_IB_EGO_VS_PIXEL_SUITE: SKILLS_IB_EGO_VS_PIXEL_STEMS,
    FINAL_10M_BASE_SKILLS_SUITE: FINAL_10M_BASE_SKILLS_STEMS,
    FINAL_10M_PQN_PIXEL_GPU0_SUITE: FINAL_10M_PQN_PIXEL_GPU0_STEMS,
    FINAL_10M_PQN_PIXEL_GPU1_SUITE: FINAL_10M_PQN_PIXEL_GPU1_STEMS,
    FINAL_10M_OVERNIGHT_GPU0_SUITE: FINAL_10M_OVERNIGHT_GPU0_STEMS,
    FINAL_10M_OVERNIGHT_GPU1_SUITE: FINAL_10M_OVERNIGHT_GPU1_STEMS,
    SPARSE_CDA_CLEAN_SUITE: SPARSE_CDA_CLEAN_STEMS,
    SPARSE_CDA_CLEAN_GPU0_SUITE: SPARSE_CDA_CLEAN_GPU0_STEMS,
    SPARSE_CDA_CLEAN_GPU1_SUITE: SPARSE_CDA_CLEAN_GPU1_STEMS,
    SPARSE_CDA_CLEAN_GPU2_SUITE: SPARSE_CDA_CLEAN_GPU2_STEMS,
    KANGAROO_SURFACEGAP_SUITE: KANGAROO_SURFACEGAP_STEMS,
    KANGAROO_FACING_SURFACEGAP_SUITE: KANGAROO_FACING_SURFACEGAP_STEMS,
    SURFACEGAP_DUAL_SUITE: SURFACEGAP_DUAL_STEMS,
    THESIS_FINAL_SUITE: THESIS_FINAL_STEMS,
    DISTBINS8_PIPELINE_SUITE: DISTBINS8_PIPELINE_STEMS,
    KANGAROO_LIFETIME_SUITE: KANGAROO_LIFETIME_STEMS,
    CLIP50_LIFETIME_PIPELINE_SUITE: CLIP50_LIFETIME_PIPELINE_STEMS,
    TIERSCALAR_PIPELINE_SUITE: TIERSCALAR_PIPELINE_STEMS,
    ANGSCALAR_ONLY_PIPELINE_SUITE: ANGSCALAR_ONLY_PIPELINE_STEMS,
    SCALAR_ABLATION_DUAL_SUITE: SCALAR_ABLATION_DUAL_STEMS,
    MINIMAL_ABLATION_SUITE: MINIMAL_ABLATION_STEMS,
    MINIMAL_ABLATION_TRAIN_SUITE: MINIMAL_ABLATION_TRAIN_STEMS,
    CLIP_LOW_PIPELINE_SUITE: CLIP_LOW_PIPELINE_STEMS,
    CLIP_SWEEP_PIPELINE_SUITE: CLIP_SWEEP_PIPELINE_STEMS,
    DISTANCE_ENCODE_PIPELINE_SUITE: DISTANCE_ENCODE_PIPELINE_STEMS,
    PIXEL_NOOUTLINE_SUITE: PIXEL_NOOUTLINE_STEMS,
    PQN_TRANSFER_SUITE: PQN_TRANSFER_STEMS,
    POV_MAPPER_SUITE: POV_MAPPER_STEMS,
    POV_NOCLIP_SUITE: POV_NOCLIP_STEMS,
    POV_NOCLIP_EVAL_SUITE: POV_NOCLIP_EVAL_STEMS,
    POV_BASELINE_SUITE: POV_BASELINE_STEMS,
    POV_BASELINE_TRAIN_SUITE: POV_BASELINE_TRAIN_STEMS,
    POV_VEL_SUITE: POV_VEL_STEMS,
    POV_PSIBINS8_SUITE: POV_PSIBINS8_STEMS,
    POV_VEL_PSIBINS8_SUITE: POV_VEL_PSIBINS8_STEMS,
    POV_FIXED2_SUITE: POV_FIXED2_STEMS,
    POV_FIXED2_TRAIN_SUITE: POV_FIXED2_TRAIN_STEMS,
    POV_FACING_SUITE: POV_FACING_STEMS,
    POV_FACING_TRAIN_SUITE: POV_FACING_TRAIN_STEMS,
    POV_FACING_COLLECT_SUITE: POV_FACING_COLLECT_STEMS,
    POV_FACING_COLLECT_TRAIN_SUITE: POV_FACING_COLLECT_TRAIN_STEMS,
    POV_FACING_V2_SUITE: POV_FACING_V2_STEMS,
    POV_FACING_V2_TRAIN_SUITE: POV_FACING_V2_TRAIN_STEMS,
    POV_FACING_V2_EVAL_SUITE: POV_FACING_V2_EVAL_STEMS,
    POV_FACING_V2_PHASE2_SUITE: POV_FACING_V2_PHASE2_STEMS,
    POV_FACING_V2_GPU0_SUITE: POV_FACING_V2_GPU0_STEMS,
    POV_FACING_V2_GPU1_SUITE: POV_FACING_V2_GPU1_STEMS,
    POV_FACING_V2_GPU2_SUITE: POV_FACING_V2_GPU2_STEMS,
    POV_FACING_ABLATION_SUITE: POV_FACING_ABLATION_STEMS,
    POV_FACING_ABLATION_GPU0_SUITE: POV_FACING_ABLATION_GPU0_STEMS,
    POV_FACING_ABLATION_GPU1_SUITE: POV_FACING_ABLATION_GPU1_STEMS,
    POV_FACING_ABLATION_GPU2_SUITE: POV_FACING_ABLATION_GPU2_STEMS,
    POV_FACING_ABLATION_PHIDEG_SUITE: POV_FACING_ABLATION_PHIDEG_STEMS,
    COLLECT_SPARSE_ARMS_SUITE: COLLECT_SPARSE_ARMS_STEMS,
    RESTRAINED_EGO_SUITE: RESTRAINED_EGO_STEMS,
}

# Short nvidia-smi / RTPT labels (1–2 tokens; project-generic, not run-specific).
SUITE_RTPT: dict[str, str] = {
    SKILL_RESHAPE_SUITE: "reshape",
    DESTROY_ABLATION_SUITE: "destroy",
    DESTROY_URGENCY_SUITE: "urgency",
    COLLECT_SHOT_PROBES_SUITE: "collect",
    EGO_POSE_PROBES_SUITE: "ego_pose",
    COLLECT_COMBAT_PROBES_SUITE: "combat",
    NO_SURFACE_DEATH_SUITE: "surface",
    EGO_CLIP_BINS_SUITE: "clip_bins",
    CLIP_BINS_NODECAY_SUITE: "nodecay",
    CLIP50_ANGBINS_FULL_SUITE: "angbins",
    THESIS_TIER_A_SUITE: "thesis",
    THESIS_FINAL_SUITE: "thesis",
    THESIS_FACING_SUITE: "thesis",
    THESIS_TIER_B_SUITE: "finetune",
    THESIS_TIER_B_FACING_SUITE: "finetune",
    THESIS_TIER_C_SUITE: "extras",
    THESIS_TIER_C_FACING_SUITE: "extras",
    NAVIGATE_SURFACE_TANDEM_PIXEL_SUITE: "nav_tandem",
    NAVIGATE_SURFACE_SURFACEGAP_SUITE: "nav_sgap",
    NAVIGATE_SURFACE_BODYGAPS_SUITE: "nav_bgaps",
    EGO_MIN_CHANNELS_SUITE: "ego_min",
    EGO_MIN_DESTROYEXTRAS_SUITE: "ego_dx",
    EGO_MIN_NOCLIP_SUITE: "ego_nc",
    EGO_MIN_NOWEAPON_OVALAVOID_SUITE: "ego_ova",
    EGO_MIN_NOWEAPON_OVALAVOID_SEEDS_SUITE: "ego_ovas",
    EGO_MIN_NOWEAPON_HEU_ABLATION_SUITE: "ego_heu",
    EGO_MIN_NOFACING_OVALAVOID_SUITE: "ego_nfova",
    EGO_MIN_NOWEAPON_SURFACESUB5_CLIPSWEEP_SUITE: "ego_clip",
    "ego_min_noweapon_ovalavoid_clipsweep": "ego_clip",
    EGO_MIN_NOWEAPON_SURFACESUB5_DISTABL_SUITE: "ego_dist",
    EGO_MIN_NOWEAPON_SURFACESUB5_DISTBANDS_SUITE: "ego_bands",
    EGO_MIN_NOWEAPON_SURFACESUB5_CLIP50_STICKY_SUITE: "ego_stick",
    EGO_MIN_NOWEAPON_SURFACESUB5_CLIPRAND_SUITE: "ego_rand",
    EGO_MIN_SURFACESUB5_CLIP10_CANFIRE_SUITE: "ego_c10cf",
    EGO_MIN_SURFACESUB5_SKILLMECH_RW_SUITE: "ego_skm",
    NEWSEAQUEST_SPAWNFIX_SUITE: "sq_new",
    ALE_PQN_MATCH_SUITE: "ale_pqn",
    PQN_PAPERHPS_OLDSEAQUEST_SUITE: "pqn_old",
    NEWSEAQUEST_CLIP10_OFFPOLICY_SUITE: "sq_c10op",
    FIXED_EGO_SKILL_BASELINE_SUITE: "ego_fix",
    EGO_MIN_SURFACESUB5_DISTPOSE_SUITE: "ego_dpose",
    EGO_MIN_SURFACESUB5_NEARTIERS_SUITE: "ego_ntier",
    EGO_MIN_SURFACESUB5_CANFIRE_OVERNIGHT_SUITE: "ego_cfire",
    ALE_CLIP10_HEU_SMOKE_SUITE: "ale_c10",
    SEAQUEST_META_OC_ABLATION_SUITE: "meta_oc",
    SEAQUEST_META_OC_SPARSE_SUITE: "meta_oc",
    SPARSE_CDA_CLIP_SWEEP_SUITE: "sparse",
    SPARSE_CDA_CLIP_TRANSFER_SUITE: "sparse_zs",
    SPARSE_CDA_CLIP_TRANSFER_GPU0_SUITE: "sparse_zs",
    SPARSE_CDA_CLIP_TRANSFER_GPU1_SUITE: "sparse_zs",
    SPARSE_CDA_CLIP_EXTREMES_SUITE: "sparse_ex",
    SPARSE_CDA_CLIP_EXTREMES_GPU0_SUITE: "sparse_ex",
    SPARSE_CDA_CLIP_EXTREMES_GPU1_SUITE: "sparse_ex",
    NOCLIP_SPARSE_SMOKE_SUITE: "noclip_sm",
    NOCLIP_SPARSE_SMOKE_GPU0_SUITE: "noclip_sm",
    NOCLIP_SPARSE_SMOKE_GPU1_SUITE: "noclip_sm",
    NOCLIP_SPARSE_SMOKE_GPU2_SUITE: "noclip_sm",
    NOCLIP_SPARSE_SMOKE_RERUN_SUITE: "noclip_sm",
    NOCLIP_SPARSE_SMOKE_AS_TP_TANDEM_SUITE: "noclip_sm",
    NOCLIP_SPARSE_SMOKE_AS_TP_SCRATCH_SUITE: "noclip_sm",
    EPS_SANITY_SUITE: "eps_san",
    HP_SWEEP_SKILLS_SUITE: "hp_sk",
    HP_SWEEP_TANDEM_SUITE: "hp_td_v2",
    HP_SWEEP_TANDEM_CKPT16_SUITE: "hp_td16",
    TANDEM_E01_SCALE_SUITE: "td_e01",
    TANDEM_E01_OC_SCALE_SUITE: "td_e01oc",
    NAVIGATE_SURFACE_CENTRES_SUITE: "nav_ctr",
    SURFACEGAP_GEOM_DUAL_SUITE: "sgap_geom",
    KANGAROO_SURFACEGAP_SUITE: "kg_sgap",
    KANGAROO_FACING_SURFACEGAP_SUITE: "kg_fsgap",
    SURFACEGAP_DUAL_SUITE: "sgap_dual",
    CLIP100_COLLECTSURFACE_PROBE_SUITE: "clip100",
    CLIP150_COLLECTSURFACE_PROBE_SUITE: "clip150",
    FACING_DISTANCE_ABLATION_SUITE: "dist_abl",
    DISTBINS8_PIPELINE_SUITE: "distbins",
    KANGAROO_LIFETIME_SUITE: "kangaroo",
    CLIP50_LIFETIME_PIPELINE_SUITE: "lifetime",
    TIERSCALAR_PIPELINE_SUITE: "tier_scalar",
    ANGSCALAR_ONLY_PIPELINE_SUITE: "ang_scalar",
    SCALAR_ABLATION_DUAL_SUITE: "scalar",
    MINIMAL_ABLATION_SUITE: "min_abl",
    MINIMAL_ABLATION_TRAIN_SUITE: "min_train",
    CLIP_LOW_PIPELINE_SUITE: "clip_low",
    CLIP_SWEEP_PIPELINE_SUITE: "clip_sweep",
    DISTANCE_ENCODE_PIPELINE_SUITE: "dist_enc",
    PIXEL_NOOUTLINE_SUITE: "nooutline",
    PQN_TRANSFER_SUITE: "pqn",
    POV_MAPPER_SUITE: "pov",
    POV_NOCLIP_SUITE: "noclip",
    POV_NOCLIP_EVAL_SUITE: "noclip",
    POV_BASELINE_SUITE: "pov_base",
    POV_BASELINE_TRAIN_SUITE: "pov_train",
    POV_VEL_SUITE: "pov_vel",
    POV_PSIBINS8_SUITE: "psi8",
    POV_VEL_PSIBINS8_SUITE: "vel_psi8",
    POV_FIXED2_SUITE: "fixed2",
    POV_FIXED2_TRAIN_SUITE: "fixed2",
    POV_FACING_SUITE: "facing",
    POV_FACING_TRAIN_SUITE: "facing",
    POV_FACING_COLLECT_SUITE: "facing",
    POV_FACING_COLLECT_TRAIN_SUITE: "facing",
    POV_FACING_V2_SUITE: "facing2",
    POV_FACING_V2_TRAIN_SUITE: "facing2",
    POV_FACING_V2_EVAL_SUITE: "facing2",
    POV_FACING_V2_PHASE2_SUITE: "facing2",
    POV_FACING_V2_GPU0_SUITE: "facing2",
    POV_FACING_V2_GPU1_SUITE: "facing2",
    POV_FACING_V2_GPU2_SUITE: "facing2",
    POV_FACING_ABLATION_SUITE: "facing_abl",
    POV_FACING_ABLATION_GPU0_SUITE: "facing_abl",
    POV_FACING_ABLATION_GPU1_SUITE: "facing_abl",
    POV_FACING_ABLATION_GPU2_SUITE: "facing_abl",
    POV_FACING_ABLATION_PHIDEG_SUITE: "facing_abl",
    SPARSE_CDA_CLEAN_SUITE: "sparse",
    SPARSE_CDA_CLEAN_GPU0_SUITE: "sparse",
    SPARSE_CDA_CLEAN_GPU1_SUITE: "sparse",
    SPARSE_CDA_CLEAN_GPU2_SUITE: "sparse",
    COLLECT_SPARSE_ARMS_SUITE: "sparse",
    RESTRAINED_EGO_SUITE: "restrain",
}


def _rtpt_label_from_suite(suite: str) -> str:
    """Derive a short cluster label from a suite slug."""
    if suite in SUITE_RTPT:
        return SUITE_RTPT[suite]
    tokens = [t for t in suite.split("_") if t not in ("suite", "train", "eval", "probe")]
    if tokens and tokens[0].startswith("clip"):
        return tokens[0]
    if len(tokens) == 1:
        return tokens[0]
    if len(tokens) >= 2:
        return f"{tokens[0]}_{tokens[1]}"
    return RTPT_BASE


def _default_batch_tag() -> str:
    """Dated suite id so W&B filters don't mix overnight batches."""
    return datetime.now().strftime("paper_eval_%Y%m%d_%H%M")


def _resolve_batch_tag(cli_value: str | None) -> str:
    if cli_value and str(cli_value).strip():
        return str(cli_value).strip()
    env = os.environ.get("PAPER_EVAL_BATCH_TAG", "").strip()
    if env:
        return env
    return _default_batch_tag()


def _attach_batch_wandb_tag(cfg: dict[str, Any], batch_tag: str) -> None:
    tags = list(cfg.get("WANDB_TAGS") or [])
    # Keep suite:paper_eval; add dated batch for grouping this overnight.
    if "suite:paper_eval" not in tags:
        tags.append("suite:paper_eval")
    batch_key = f"batch:{batch_tag}"
    if batch_key not in tags:
        tags.append(batch_key)
    cfg["WANDB_TAGS"] = tags
    cfg["PAPER_EVAL_BATCH_TAG"] = batch_tag

# Representative smoke matrix (every KIND / PHASE exercised once).
SMOKE_STEMS = (
    # RQ1 train
    "seaquest_skills_ego",
    "seaquest_pqn",
    # RQ1 eval
    "seaquest_skills_ego_eval",
    "seaquest_heuristic_random",
    "seaquest_pqn_eval",
    # RQ2
    "pong_transfer_ego",
    "pong_finetune_ego",
    "pong_from_scratch_ego",
    # RQ3
    "seaquest_meta_pixel_ego_skills",
    "seaquest_tandem_pixel_ego",
)

RQ1_SKILL_STEMS = (
    "seaquest_skills_ego_clip50_angbins",
    "seaquest_skills_pixel",
    "seaquest_skills_oc",
)


def _prepend_jaxatari_root(root: Path | None) -> None:
    if root is None:
        return
    src = root / "src"
    if not src.is_dir():
        raise SystemExit(f"JAXAtari root missing src/: {root}")
    sys.path.insert(0, str(src))
    os.environ["PAPER_EVAL_JAXATARI_ROOT"] = str(root)
    print(f"Using JAXAtari from {src}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _flatten_experiment_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    alg = raw.pop("alg", {}) or {}
    return {**raw, **alg}


def _parse_shard(spec: str) -> tuple[int, int]:
    parts = spec.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid shard {spec!r}; expected INDEX/TOTAL")
    shard_one_based, num_shards = int(parts[0]), int(parts[1])
    if num_shards < 1 or not 1 <= shard_one_based <= num_shards:
        raise ValueError(f"Invalid shard {spec!r}")
    return shard_one_based - 1, num_shards


def _is_runnable_config(path: Path) -> bool:
    if path.name.startswith("_") or path.name.startswith("manifest"):
        return False
    try:
        path.relative_to(MODS_DIR)
        return False
    except ValueError:
        pass
    return path.suffix in {".yaml", ".yml"}


def _discover_configs(
    *,
    rq: int | None = None,
    only: list[str] | None = None,
    phase: str | None = None,
) -> list[Path]:
    paths = sorted(p for p in CONFIG_DIR.rglob("*.yaml") if _is_runnable_config(p))
    if rq is not None:
        paths = [p for p in paths if int(_load_yaml(p).get("PAPER_EVAL_RQ", -1)) == rq]
    if phase is not None:
        paths = [
            p
            for p in paths
            if str(_load_yaml(p).get("PAPER_EVAL_PHASE", "train")).lower() == phase.lower()
        ]
    if only:
        wanted = {name.removesuffix(".yaml") for name in only}
        paths = [p for p in paths if p.stem in wanted]
        missing = wanted - {p.stem for p in paths}
        if missing:
            raise FileNotFoundError(f"Config(s) not found: {sorted(missing)}")
    return paths


def _shard_paths(paths: list[Path], *, shard_idx: int, num_shards: int) -> list[Path]:
    return [p for i, p in enumerate(paths) if i % num_shards == shard_idx]


def _rtpt_name(
    shard: str | None,
    rq: int | None,
    *,
    suite: str | None = None,
) -> str:
    if suite:
        base = _rtpt_label_from_suite(suite)
    elif rq is not None:
        base = f"rq{rq}"
    else:
        base = RTPT_BASE
    if base != RTPT_BASE and not base.startswith(f"{RTPT_BASE}_"):
        base = f"{RTPT_BASE}_{base}"
    if shard is None:
        return base
    return f"{base}_s{_parse_shard(shard)[0] + 1}"


def _align_trial_timesteps(cfg: dict[str, Any]) -> None:
    batch = int(cfg.get("NUM_ENVS", 128)) * int(cfg.get("NUM_STEPS", 128))
    if batch <= 0:
        return
    total = int(cfg.get("TOTAL_TIMESTEPS", 0))
    cfg["TOTAL_TIMESTEPS"] = max(1, total // batch) * batch
    if "TOTAL_TIMESTEPS_DECAY" in cfg:
        cfg["TOTAL_TIMESTEPS_DECAY"] = cfg["TOTAL_TIMESTEPS"]


def _configure_lean_disk() -> Path:
    """Put wandb local files on $TMPDIR (often a separate volume from /home)."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
    tmp_root = Path(os.environ.get("TMPDIR") or "/tmp") / user / "cbl_paper_eval"
    wandb_dir = tmp_root / "wandb"
    cache_dir = tmp_root / "wandb_cache"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
    # Avoid large local artifact media if anything slips through.
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    try:
        tmp_free = shutil.disk_usage(tmp_root).free / (1024.0**3)
        tmp_note = f"{tmp_free:.1f} GiB free on tmp"
    except OSError:
        tmp_note = "tmp free unknown"
    print(
        f"lean disk: WANDB_DIR={wandb_dir}  WANDB_CACHE_DIR={cache_dir}  ({tmp_note})",
        flush=True,
    )
    return tmp_root


def _is_from_scratch_train(raw: dict[str, Any], *, stem: str = "") -> bool:
    """True for base from-scratch train YAMLs (not ZS/finetune/transfer/eval).

    Used to gate end-of-train wandb videos: only scratch trains get RECORD_VIDEO
    when ``--video`` is set, so review stays cheap and focused on base behavior.
    """
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    if phase != "train":
        return False
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    if kind in {
        "transfer",
        "heuristic_random",
        "fixed_skills",
        "fixed_ego_rules",
        "constant_ego",
    }:
        return False
    name = (stem or str(raw.get("WANDB_RUN_NAME", ""))).lower()
    if "from_scratch" in name or "from_scratch" in stem.lower():
        return True
    # Fallback: train with no upstream skill/meta init and empty requires.
    if raw.get("INIT_MODEL_PATH") or raw.get("SKILLS_MODEL_PATH"):
        return False
    requires = raw.get("PAPER_EVAL_REQUIRES") or []
    if isinstance(requires, (list, tuple)) and len(requires) > 0:
        return False
    return True


def _apply_run_defaults(
    cfg: dict[str, Any],
    *,
    wandb_mode: str | None,
    trial_steps: int | None,
    record_video: bool,
    seed: int | None,
    batch_tag: str | None = None,
    save_training_metrics_csv: bool = True,
) -> None:
    if wandb_mode is not None:
        cfg["WANDB_MODE"] = wandb_mode
    if trial_steps is not None:
        cfg["TOTAL_TIMESTEPS"] = int(trial_steps)
        cfg.setdefault("WANDB_TAGS", [])
        if "trial" not in cfg["WANDB_TAGS"]:
            cfg["WANDB_TAGS"] = [*cfg["WANDB_TAGS"], "trial"]
        _align_trial_timesteps(cfg)
    if seed is not None:
        cfg["SEED"] = int(seed)
    cfg.setdefault("ENTITY", "paul-seitz-tu-darmstadt")
    cfg.setdefault("PROJECT", "concept_based_learning")
    cfg.setdefault("SEED", DEFAULT_SEED)
    cfg.setdefault("NUM_SEEDS", 1)
    cfg["RECORD_VIDEO"] = record_video
    cfg["VIDEO_AT_END"] = record_video
    cfg["SAVE_TRAINING_METRICS_CSV"] = bool(save_training_metrics_csv)
    if batch_tag:
        _attach_batch_wandb_tag(cfg, batch_tag)


def _skill_ckpt_from_save(save_path: str | Path, env_name: str, seed: int) -> Path:
    root = Path(save_path)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / env_name / f"hierarchical_pqn_{env_name}_seed{seed}.safetensors"


def _config_yaml_for_stem(stem: str) -> Path | None:
    """Resolve a paper_eval YAML by stem (rq1 first, then any rq/)."""
    rq1 = CONFIG_DIR / "rq1" / f"{stem}.yaml"
    if rq1.is_file():
        return rq1
    matches = list(CONFIG_DIR.rglob(f"{stem}.yaml"))
    return matches[0] if matches else None


def _eval_json_path(stem: str) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "models"
        / "paper_eval"
        / "eval"
        / f"{stem}_mods.json"
    )


def _flat_pqn_ckpt_from_cfg(cfg: dict[str, Any], *, seed: int) -> Path:
    from concept_based_learning.paper_eval.checkpoints import resolve_flat_pqn_checkpoint

    return resolve_flat_pqn_checkpoint(
        object_centric=bool(cfg.get("OBJECT_CENTRIC", True)),
        env_name=str(cfg.get("ENV_NAME", "seaquest")).lower(),
        seed=int(cfg.get("SEED", seed)),
        save_root=cfg.get("SAVE_PATH"),
        vmap_index=0,
        oc_input_mode=str(cfg.get("OC_INPUT_MODE") or "").strip() or None,
    )


def _skills_train_artifacts(cfg: dict[str, Any], *, seed: int, kind: str) -> list[Path]:
    """Expected checkpoint files produced by a skills / meta / tandem train."""
    env_name = str(cfg.get("ENV_NAME", "seaquest")).lower()
    seed_i = int(cfg.get("SEED", seed))
    save_root = cfg.get("SAVE_PATH", "./outputs/models")
    root = Path(save_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    env_dir = root / env_name
    alg = str(cfg.get("ALG_NAME", "hierarchical_pqn"))
    if kind in {"meta", "tandem"}:
        return [
            env_dir / f"{alg}_{env_name}_seed{seed_i}_skills.safetensors",
            env_dir / f"{alg}_{env_name}_seed{seed_i}_meta.safetensors",
        ]
    # skills_only / finetune: single stacked skill ckpt (no _skills suffix)
    return [env_dir / f"{alg}_{env_name}_seed{seed_i}.safetensors"]


def _eval_json_complete(
    path: Path,
    *,
    env_name: str | None = None,
    expected_mod_keys: list[str] | None = None,
) -> tuple[bool, str]:
    """
    True only if the mods JSON exists, parses, and has a non-empty ``results``
    map that covers expected mod keys (at least ``base``).

    Partial writes from earlier disk-full failures are treated as incomplete.
    """
    if not path.is_file() or path.stat().st_size < 8:
        return False, f"missing {path}"
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"unreadable {path.name}: {exc}"
    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        return False, f"empty results in {path.name}"
    if "base" not in results:
        return False, f"{path.name} missing results['base']"

    keys = expected_mod_keys
    if keys is None and env_name:
        try:
            from concept_based_learning.paper_eval.mods import mod_specs_for_env

            keys = [m.key for m in mod_specs_for_env(env_name)]
        except Exception:
            keys = None
    if keys:
        missing = [k for k in keys if k not in results]
        if missing:
            return (
                False,
                f"{path.name} incomplete ({len(results)}/{len(keys)} mods; "
                f"missing e.g. {missing[:3]})",
            )
    return True, f"{path.name} ({len(results)} mods)"


def experiment_done(path: Path, *, seed: int) -> tuple[bool, str]:
    """
    Return (done, reason) based on on-disk artifacts.

    Train → checkpoint(s) present.
    Eval / transfer / heuristic → complete mods JSON (all allowlisted mods).
    """
    raw = _load_yaml(path)
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    status = str(raw.get("PAPER_EVAL_STATUS", "ready")).lower()
    if status in {"stub", "planned"}:
        return True, f"status={status} (skipped)"

    if phase == "eval" or kind in {
        "heuristic_random",
        "transfer",
        "fixed_skills",
        "fixed_ego_rules",
        "constant_ego",
    }:
        env_name = str(raw.get("ENV_NAME") or raw.get("PAPER_EVAL_MODS") or "seaquest")
        if isinstance(env_name, list):
            env_name = env_name[0] if env_name else "seaquest"
        env_name = str(env_name).lower()
        mods_allowlist = str(raw.get("PAPER_EVAL_MODS") or env_name).lower()
        out = _eval_json_path(path.stem)
        expected = (
            list(raw["PAPER_EVAL_MOD_KEYS"])
            if raw.get("PAPER_EVAL_MOD_KEYS")
            else None
        )
        ok, detail = _eval_json_complete(
            out, env_name=mods_allowlist, expected_mod_keys=expected
        )
        if ok:
            return True, f"eval {detail}"
        return False, f"eval incomplete: {detail}"

    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    if kind == "flat_pqn":
        ckpt = _flat_pqn_ckpt_from_cfg(cfg, seed=seed)
        if ckpt.is_file():
            return True, f"ckpt {ckpt.name}"
        return False, f"missing {ckpt}"

    arts = _skills_train_artifacts(cfg, seed=seed, kind=kind)
    missing = [p for p in arts if not p.is_file()]
    if not missing:
        return True, "ckpt " + ", ".join(p.name for p in arts)
    return False, "missing " + ", ".join(str(p) for p in missing)


def _filter_remaining_paths(
    paths: list[Path],
    *,
    seed: int,
    exclude: set[str],
    skip_done: bool,
) -> list[Path]:
    """Drop completed runs and explicitly excluded stems (+ dependents)."""
    exclude_l = {s.removesuffix(".yaml") for s in exclude}
    remaining: list[Path] = []
    skipped_done = 0
    skipped_excl = 0
    skipped_dep = 0
    for path in paths:
        stem = path.stem
        if stem in exclude_l:
            print(f"  SKIP (excluded): {stem}", flush=True)
            skipped_excl += 1
            continue
        requires = _required_rq1_stems(path)
        blocked = [r for r in requires if r in exclude_l]
        if blocked:
            print(
                f"  SKIP (depends on excluded {', '.join(blocked)}): {stem}",
                flush=True,
            )
            skipped_dep += 1
            continue
        if skip_done:
            done, reason = experiment_done(path, seed=seed)
            if done:
                print(f"  SKIP (done: {reason}): {stem}", flush=True)
                skipped_done += 1
                continue
        remaining.append(path)
    print(
        f"resume filter: keep {len(remaining)} / {len(paths)} "
        f"(done={skipped_done}, excluded={skipped_excl}, dep-excluded={skipped_dep})",
        flush=True,
    )
    return remaining


def _rq1_checkpoint_ready(stem: str, *, seed: int) -> tuple[bool, Path]:
    """True when the train artifacts for ``stem`` exist (skills, or tandem pair)."""
    path = _config_yaml_for_stem(stem)
    if path is None:
        return False, CONFIG_DIR / "rq1" / f"{stem}.yaml"
    raw = _load_yaml(path)
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    seed_i = int(cfg.get("SEED", seed))
    if kind == "flat_pqn":
        ckpt = _flat_pqn_ckpt_from_cfg(cfg, seed=seed_i)
        return ckpt.is_file(), ckpt
    artifacts = _skills_train_artifacts(
        cfg, seed=seed_i, kind=kind
    )
    missing = [p for p in artifacts if not p.is_file()]
    if missing:
        return False, missing[0]
    return True, artifacts[0]


def _check_rq_gate(paths: list[Path], *, seed: int) -> None:
    missing: list[str] = []
    for path in paths:
        raw = _load_yaml(path)
        # Explicit PAPER_EVAL_REQUIRES_RQ1 (even []) wins — from-scratch tandem
        # sets [] and must not inherit the legacy seaquest_skills_ego wait.
        if "PAPER_EVAL_REQUIRES_RQ1" in raw:
            requires = list(raw.get("PAPER_EVAL_REQUIRES_RQ1") or [])
        else:
            requires = []
            rq = int(raw.get("PAPER_EVAL_RQ", 1))
            kind = str(raw.get("PAPER_EVAL_KIND", "")).lower()
            phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
            if not requires and rq >= 2 and phase != "train":
                requires = list(RQ1_SKILL_STEMS[:1])
            if kind in {"meta", "tandem"} and not requires:
                requires = ["seaquest_skills_ego"]
        for stem in requires:
            ok, ckpt = _rq1_checkpoint_ready(str(stem), seed=seed)
            if not ok:
                missing.append(f"{path.stem} requires {stem} → missing {ckpt}")
    if missing:
        raise SystemExit(
            "RQ gate failed (train RQ1 first, or pass --skip-rq-gate):\n  - "
            + "\n  - ".join(missing)
        )


def _required_rq1_stems(path: Path) -> list[str]:
    raw = _load_yaml(path)
    if "PAPER_EVAL_REQUIRES_RQ1" in raw:
        return [str(s) for s in (raw.get("PAPER_EVAL_REQUIRES_RQ1") or [])]
    requires: list[str] = []
    rq = int(raw.get("PAPER_EVAL_RQ", 1))
    kind = str(raw.get("PAPER_EVAL_KIND", "")).lower()
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    if not requires and rq >= 2 and phase != "train":
        requires = list(RQ1_SKILL_STEMS[:1])
    if kind in {"meta", "tandem"} and not requires:
        requires = ["seaquest_skills_ego"]
    return [str(s) for s in requires]


def _is_no_space_error(exc: BaseException) -> bool:
    """True for ENOSPC and common chained / string forms of disk-full failures."""
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, OSError) and cur.errno == errno.ENOSPC:
            return True
        msg = str(cur).lower()
        if "no space left on device" in msg or "[errno 28]" in msg:
            return True
        if cur.__cause__ is not None:
            stack.append(cur.__cause__)
        if cur.__context__ is not None and cur.__context__ is not cur.__cause__:
            stack.append(cur.__context__)
    return False


def _free_gb(path: Path | None = None) -> float:
    """Free space (GiB) on the filesystem that holds ``path`` (default: project root)."""
    target = path if path is not None else PROJECT_ROOT
    try:
        usage = shutil.disk_usage(target)
    except FileNotFoundError:
        usage = shutil.disk_usage(PROJECT_ROOT)
    return float(usage.free) / (1024.0**3)


def _assert_disk_space(
    *,
    min_free_gb: float,
    where: str,
    path: Path | None = None,
) -> None:
    """Crash immediately if the project volume is too full to continue safely."""
    if min_free_gb <= 0:
        return
    free = _free_gb(path)
    if free < float(min_free_gb):
        raise SystemExit(
            f"Aborting ({where}): only {free:.2f} GiB free on "
            f"{path or PROJECT_ROOT} (need >= {min_free_gb:.2f} GiB). "
            "Free disk space (e.g. wandb/run-* , old outputs) and rerun."
        )


def _wait_for_path_deps(
    path: Path,
    *,
    seed: int,
    poll_s: float = 60.0,
    timeout_h: float = 36.0,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    failed_local_stems: set[str] | None = None,
) -> None:
    """Block until PAPER_EVAL_REQUIRES_RQ1 checkpoints exist (multi-GPU overnight)."""
    requires = _required_rq1_stems(path)
    if not requires:
        return
    failed_local = failed_local_stems or set()
    already_failed = [s for s in requires if s in failed_local]
    if already_failed:
        raise SystemExit(
            f"Aborting wait for {path.stem}: required dep(s) already failed in "
            f"this worker (will never appear): {', '.join(already_failed)}"
        )
    deadline = time.time() + max(60.0, timeout_h * 3600.0)
    last_msg = ""
    while True:
        _assert_disk_space(
            min_free_gb=min_free_gb,
            where=f"wait-deps for {path.stem}",
        )
        missing: list[str] = []
        for stem in requires:
            if stem in failed_local:
                raise SystemExit(
                    f"Aborting wait for {path.stem}: required dep {stem} failed "
                    "in this worker while waiting."
                )
            ok, ckpt = _rq1_checkpoint_ready(stem, seed=seed)
            if not ok:
                missing.append(f"{stem} → {ckpt}")
        if not missing:
            print(f"  Deps ready for {path.stem}", flush=True)
            return
        msg = f"  Waiting on deps for {path.stem}: {', '.join(missing)}"
        if msg != last_msg:
            print(msg, flush=True)
            last_msg = msg
        if time.time() >= deadline:
            raise SystemExit(
                f"Timed out after {timeout_h}h waiting for deps of {path.stem}:\n  - "
                + "\n  - ".join(missing)
            )
        time.sleep(poll_s)


_PHASE_SORT = {"train": 0, "finetune": 1, "eval": 2}


def _schedule_key(path: Path) -> tuple:
    """Prefer ungated RQ1 trains first so other GPUs can unblock sooner."""
    raw = _load_yaml(path)
    requires = _required_rq1_stems(path)
    rq = int(raw.get("PAPER_EVAL_RQ", 1))
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    return (1 if requires else 0, rq, _PHASE_SORT.get(phase, 9), path.stem)


def _launch_multi_gpu(argv: list[str], gpu_ids: list[str]) -> int:
    """
    Re-exec this script once per GPU with ``--shard i/N --wait-deps``.

    Parent blocks until all workers exit. Returns max worker return code.
    """
    n = len(gpu_ids)
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--gpus":
            i += 2  # skip flag + value
            continue
        if a.startswith("--gpus="):
            i += 1
            continue
        # Workers get their own --shard; drop a parent-provided one if any.
        if a == "--shard":
            i += 2
            continue
        if a.startswith("--shard="):
            i += 1
            continue
        cleaned.append(a)
        i += 1

    if "--wait-deps" not in cleaned:
        cleaned.append("--wait-deps")

    procs: list[subprocess.Popen] = []
    for wi, gpu in enumerate(gpu_ids):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            *cleaned,
            "--shard",
            f"{wi + 1}/{n}",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        # Share one batch tag across workers (parent sets PAPER_EVAL_BATCH_TAG).
        if "PAPER_EVAL_BATCH_TAG" not in env:
            env["PAPER_EVAL_BATCH_TAG"] = _default_batch_tag()
        print(f"\n=== multi-GPU worker {wi + 1}/{n}  CUDA_VISIBLE_DEVICES={gpu} ===")
        print(" ", " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(PROJECT_ROOT)))

    codes = [p.wait() for p in procs]
    worst = max(codes) if codes else 1
    if worst != 0:
        print(f"\nmulti-GPU: worker exit codes {codes}", flush=True)
    else:
        print("\nmulti-GPU: all workers finished successfully.", flush=True)
    return int(worst)


def _n_eval_mods(raw: dict[str, Any], *, trial: bool) -> int:
    env = str(raw.get("ENV_NAME", "seaquest"))
    try:
        from concept_based_learning.paper_eval.mods import mod_specs_for_env

        return max(1, len(mod_specs_for_env(env, trial=trial)))
    except Exception:
        return 1


def _updates_for_config_path(
    path: Path, *, trial_steps: int | None, trial: bool = False
) -> int:
    """Weighted RTPT units for one config (pixel train ≫ ego; evals are non-zero)."""
    from concept_based_learning.training.rtpt_progress import (
        RTPT_EVAL_UNITS_PER_MOD,
        num_updates_from_timesteps,
        rtpt_weight_for_config,
    )

    raw = _load_yaml(path)
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    if phase == "eval" or kind == "heuristic_random":
        return _n_eval_mods(raw, trial=trial) * RTPT_EVAL_UNITS_PER_MOD
    if trial_steps is not None:
        cfg["TOTAL_TIMESTEPS"] = int(trial_steps)
        _align_trial_timesteps(cfg)
    updates = num_updates_from_timesteps(
        int(cfg.get("TOTAL_TIMESTEPS", 0)),
        num_steps=int(cfg.get("NUM_STEPS", 128)),
        num_envs=int(cfg.get("NUM_ENVS", 128)),
    )
    return updates * rtpt_weight_for_config(cfg, kind=kind)


def _bind_rtpt_job(path: Path) -> None:
    """Scale jax.debug.callback ticks for the job about to run (evals tick per mod)."""
    from concept_based_learning.training.rtpt_progress import (
        rtpt_weight_for_config,
        set_rtpt_job_weight,
    )

    raw = _load_yaml(path)
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    if phase == "eval" or kind == "heuristic_random":
        set_rtpt_job_weight(1)
        return
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    weight = rtpt_weight_for_config(cfg, kind=kind)
    set_rtpt_job_weight(weight)
    if weight > 1:
        print(f"  RTPT job weight={weight} (pixel/oc wall-time scaling)", flush=True)


def _init_rtpt_for_paths(
    paths: list[Path],
    *,
    experiment_name: str,
    trial_steps: int | None,
    trial: bool = False,
) -> None:
    from concept_based_learning.training.rtpt_progress import start_rtpt

    total = max(
        1,
        sum(
            _updates_for_config_path(p, trial_steps=trial_steps, trial=trial)
            for p in paths
        ),
    )
    if start_rtpt(
        {
            "RTPT_ENABLED": True,
            "RTPT_NAME_INITIALS": "PS",
            "RTPT_EXPERIMENT_NAME": experiment_name,
            "RTPT_STEP_EVERY": 10,
            "TOTAL_TIMESTEPS": total,
            "NUM_STEPS": 1,
            "NUM_ENVS": 1,
        }
    ):
        print(
            f"RTPT initialized: {total:,} weighted units across {len(paths)} run(s) "
            f"(evals included; pixel×10, oc×2)\n"
        )


def _visible_gpu_indices() -> list[int] | None:
    """Physical GPU ids this process may use, or None if unrestricted."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    if raw == "-1":
        return []
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
    return ids


def _gpu_mem_brief() -> str:
    """One-line nvidia-smi used/total for *this process's* visible GPUs."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        visible = _visible_gpu_indices()
        if visible is not None:
            if not visible:
                return "(cpu / no visible gpu)"
            cmd.extend(["-i", ",".join(str(i) for i in visible)])
        out = subprocess.check_output(cmd, text=True, timeout=5)
        parts = []
        for line in out.strip().splitlines():
            idx, used, total = [p.strip() for p in line.split(",")]
            parts.append(f"gpu{idx}:{used}/{total}MiB")
        return " ".join(parts) if parts else "(no gpu)"
    except Exception:
        return "(nvidia-smi unavailable)"


def _pin_jaxatari_action_sets_to_host() -> None:
    """
    Copy JAXAtari ``ACTION_SET`` class attrs to host numpy.

    Those sets are module-level ``jnp.array``s. Deleting live device buffers
    (or ``jax.clear_backends``) tombstones them, and the next ``AtariWrapper``
    then fails with ``Array has been deleted with shape=int32[18]``.
    """
    try:
        import inspect

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jaxatari.core import GAME_MODULES
        from jaxatari.environment import JaxEnvironment
    except Exception:
        return

    for module_path in GAME_MODULES.values():
        try:
            module = __import__(module_path, fromlist=["*"])
        except Exception:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, JaxEnvironment) or obj is JaxEnvironment:
                continue
            arr = getattr(obj, "ACTION_SET", None)
            if arr is None:
                continue
            try:
                if isinstance(arr, jax.Array) and getattr(arr, "is_deleted", lambda: False)():
                    continue
                host = np.asarray(arr, dtype=np.int32)
            except Exception:
                continue
            # Numpy stays off-device; jnp.take / np.array still accept it.
            try:
                obj.ACTION_SET = host
            except Exception:
                try:
                    obj.ACTION_SET = jnp.asarray(host, dtype=jnp.int32)
                except Exception:
                    pass


def _clear_jax_memory() -> None:
    """Drop compile caches. Do not delete live buffers or the JAX backend.

    Overnight trains used ``live_buffers().delete()`` / ``clear_backends()`` to
    reclaim VRAM. That also tombstones JAXAtari class constants (``ACTION_SET``),
    so the next eval in the same process crashes in ``AtariWrapper.__init__``.
    ``clear_caches`` + GC is enough for eval→eval; the CUDA pool is reused.
    """
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass


def _release(*, label: str = "") -> None:
    """Best-effort VRAM reclaim between paper_eval jobs in the same process."""
    before = _gpu_mem_brief()
    _pin_jaxatari_action_sets_to_host()
    gc.collect()
    _clear_jax_memory()
    gc.collect()
    _pin_jaxatari_action_sets_to_host()
    after = _gpu_mem_brief()
    tag = f" after {label}" if label else ""
    print(f"  GPU release{tag}: {before} → {after}", flush=True)


def _print_mods(env: str | None) -> None:
    from concept_based_learning.paper_eval.mods import load_mod_allowlist, mod_specs_for_env

    envs = [env] if env else sorted(p.stem for p in MODS_DIR.glob("*.yaml"))
    for name in envs:
        raw = load_mod_allowlist(name)
        print(f"\n=== mods/{name}.yaml  status={raw.get('status', '?')} ===")
        for spec in mod_specs_for_env(name):
            mods_s = ",".join(spec.mods) if spec.mods else "(base)"
            print(f"  {spec.key:28s} [{spec.category:10s}]  {mods_s}")


def _print_matrix(paths: list[Path]) -> None:
    print(f"\n=== paper_eval ({CONFIG_DIR.relative_to(PROJECT_ROOT)}) ===")
    print(f"{'Config':<36} {'RQ':>2} {'phase':<8} {'kind':<16} status")
    print("-" * 90)
    for path in paths:
        raw = _load_yaml(path)
        print(
            f"{path.stem:<36} "
            f"{str(raw.get('PAPER_EVAL_RQ', '?')):>2} "
            f"{str(raw.get('PAPER_EVAL_PHASE', 'train')):<8} "
            f"{str(raw.get('PAPER_EVAL_KIND', 'skills')):<16} "
            f"{raw.get('PAPER_EVAL_STATUS', 'ready')}"
        )
    print(f"\n{len(paths)} configs. W&B filter: suite:paper_eval")


# Eval YAMLs may set PAPER_EVAL_ARCHITECTURE_OF to pin network/obs shape to a
# train config while still overriding policy/eval knobs locally.
_EVAL_OVERRIDE_KEYS = (
    # Critical for transfer: architecture YAMLs are Seaquest trains; without these
    # the eval silently keeps ENV_NAME=seaquest (mislabeled Pong/BeamRider scores).
    "ENV_NAME",
    "MAX_OBJECTS",
    "SAVE_PATH",
    "WANDB_RUN_NAME",
    "WANDB_TAGS",
    "HEURISTIC_MODE",
    "NO_TARGET_GAME_ACTION",
    "SKILL_DEATH_PENALIZE_COLLECT",
    "COLLECT_SPARSE_ONLY",
    "COLLECT_LIVING_PENALTY",
    "DESTROY_CONTACT_FAIL",
    "DESTROY_LIVING_PENALTY",
    "DESTROY_Y_ALIGN",
    "DESTROY_STANDOFF",
    "EVAL_GAME_MODS",
    "GAME_MODS",
    "SEAQUEST_SWAP_DIVER_ENEMY_LABELS",
    "SEAQUEST_SWAP_DIVER_SHARK_ROLES",
    "SURFACE_DANGER_CLOSE_RANGE",
    "DESTROY_INTERRUPT_RANGE",
    "AVOID_INTERRUPT_RANGE",
    "AVOID_INTERRUPT_RANGE_Y",
    "SURFACE_SUB_AVOID_RANGE_X",
    "SURFACE_SUB_AVOID_RANGE_Y",
    "SURFACE_SUB_NEAR_RANGE",
    "DIVER_LANE_Y_PX",
    "DIVER_LANE_BLOCK_RANGE",
    "COLLECT_REQUIRES_NO_DESTROY",
    "COLLECT_EDGE_MARGIN_PX",
    "COLLECT_EDGE_HARD",
    "COLLECT_REQUIRE_FULLY_ON_SCREEN",
    "COLLECT_SKIP_WHEN_FULL",
    "COLLECT_NEAREST_IN_RANGE_ONLY",
    "COLLECT_NEAR_RANGE",
    "ENEMY_REQUIRE_ON_SCREEN",
    "DESTROY_NEAREST_IN_DANGER_ONLY",
    # Ego layout must follow the transfer YAML / ckpt (scalar snap flags included).
    "USE_LEGACY_EGO_OBS",
    "USE_EGO_SIZE_FEATURES",
    "USE_EGO_VELOCITY_FEATURES",
    "USE_EGO_BLOCKING_FEATURES",
    "USE_EGO_WEAPON_FEATURES",
    "USE_EGO_FACING_ENEMY_FEATURES",
    "EGO_FORCE_CAN_FIRE",
    "USE_EGO_SHOT_FEATURES",
    "USE_EGO_SHOT_BETWEEN",
    "EGO_DISTANCE_CLIP_MAX_PX",
    "EGO_DISTANCE_CLIP_CHOICES_PX",
    "EGO_DISTANCE_MODE",
    "EGO_DISTANCE_DELTA_SIGN_EPS_PX",
    "EGO_ANGLE_BINS",
    "EGO_PSI_ANGLE_BINS",
    "EGO_PSI_ANGLE_SCALAR",
    "EGO_ANGLE_BINS_SCALAR",
    "EGO_ANGLE_SINCOS_SNAP",
    "EGO_DISTANCE_BINS",
    "EGO_DISTANCE_BINS_SCALAR",
    "EGO_DISTANCE_BIN_CENTERS_PX",
    "EGO_DISTANCE_BIN_MAX_PX",
    "EGO_DISTANCE_FLAG_THRESHOLDS_PX",
    "EGO_DISTANCE_HARD_EDGES_PX",
    "EGO_RESTRAINED_POSE",
    "EGO_RESTRAINED_LATERAL_BAND_PX",
    "EGO_RESTRAINED_NEAR_FAR_PX",
    "USE_EGO_HIT_CORRIDOR",
    "EGO_HIT_CORRIDOR_SAT",
    "EGO_AABB_GEOM",
    "EGO_LEGAL_ACTION_MASK",
    "EGO_LEGAL_ACTION_MASK_MODE",
    # Pixel ablation: keep outline on/off from the eval YAML when architecture
    # is pinned to a train config that still paints by default.
    "SKILL_PIXEL_PAINT_TARGET_OUTLINE",
    "USE_POV_RELATIVE_MAPPER",
    "USE_POV_FIXED_BILATERAL",
    "POV_BILATERAL_AXIS",
    "USE_POV_FACING_FRAME",
    "EGO_POSE_XY",
    "EGO_ACTION_DIM",
    "SKILL_POLICY_MODE",
    "FIXED_EGO_ALIGN_DEG",
    "SEED",
    "EVAL_SEED",
    "EVAL_NUM_SEEDS",
    "OFF_POLICY",
)


def _architecture_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Load arch YAML referenced by PAPER_EVAL_ARCHITECTURE_OF if present.

    When an architecture ref is used, still apply eval-local overrides from
    ``raw`` (e.g. ``HEURISTIC_MODE: avoid_only`` on an AVOID probe).
    """
    ref = raw.get("PAPER_EVAL_ARCHITECTURE_OF")
    if not ref:
        return _flatten_experiment_cfg(copy.deepcopy(raw))
    ref_path = CONFIG_DIR / "rq1" / f"{str(ref).removesuffix('.yaml')}.yaml"
    if not ref_path.is_file():
        # search
        matches = list(CONFIG_DIR.rglob(f"{str(ref).removesuffix('.yaml')}.yaml"))
        if not matches:
            raise FileNotFoundError(f"Architecture config not found: {ref}")
        ref_path = matches[0]
    cfg = _flatten_experiment_cfg(copy.deepcopy(_load_yaml(ref_path)))
    eval_flat = _flatten_experiment_cfg(copy.deepcopy(raw))
    for key in _EVAL_OVERRIDE_KEYS:
        if key in eval_flat:
            cfg[key] = eval_flat[key]
    return cfg


def _run_skills_train(path: Path, **kwargs) -> None:
    import main as train_main

    raw = _load_yaml(path)
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    _apply_run_defaults(cfg, **kwargs)
    print(f"\n{'=' * 70}\n  TRAIN skills: {path.relative_to(CONFIG_DIR)}")
    print(
        f"  ENV={cfg.get('ENV_NAME')} MODE={cfg.get('TRAIN_MODE')} "
        f"INPUT={cfg.get('SKILL_INPUT_MODE')} STEPS={cfg.get('TOTAL_TIMESTEPS', 0):,}\n"
        f"{'=' * 70}\n"
    )
    try:
        train_main.single_run(cfg)
    finally:
        _release(label=path.stem)


def _run_flat_pqn_train(path: Path, **kwargs) -> None:
    from concept_based_learning.baselines.flat_pqn_train import train_flat_pqn

    raw = _load_yaml(path)
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    _apply_run_defaults(cfg, **kwargs)
    print(f"\n{'=' * 70}\n  TRAIN flat_pqn: {path.relative_to(CONFIG_DIR)}")
    print(
        f"  ENV={cfg.get('ENV_NAME')} OC={cfg.get('OBJECT_CENTRIC')} "
        f"STEPS={cfg.get('TOTAL_TIMESTEPS', 0):,}\n{'=' * 70}\n"
    )
    try:
        train_flat_pqn(cfg)
    finally:
        _release(label=path.stem)


def _run_mod_eval(
    path: Path,
    *,
    trial: bool,
    seed: int | None,
    eval_episodes: int,
    eval_max_frames: int,
    wandb_mode: str | None = None,
) -> None:
    from concept_based_learning.paper_eval.eval_harness import run_mod_eval

    raw = _load_yaml(path)
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    env_name = str(raw.get("ENV_NAME", "seaquest"))
    cfg = _architecture_config(raw)
    if seed is not None:
        cfg["SEED"] = int(seed)
    cfg.setdefault("SEED", DEFAULT_SEED)
    train_seed = int(cfg["SEED"])
    # Eval episode RNGs must not reuse the training seed (avoids lucky-seed
    # overfitting on Seaquest). Prefer explicit EVAL_SEED; else train_seed+10000.
    if "EVAL_SEED" in raw:
        eval_base_seed = int(raw["EVAL_SEED"])
    elif "EVAL_SEED" in cfg:
        eval_base_seed = int(cfg["EVAL_SEED"])
    else:
        eval_base_seed = train_seed + 10_000
    eval_num_seeds = int(
        raw.get("EVAL_NUM_SEEDS", cfg.get("EVAL_NUM_SEEDS", raw.get("NUM_SEEDS", 1)))
    )
    if eval_num_seeds < 1:
        eval_num_seeds = 1
    if eval_base_seed == train_seed:
        raise SystemExit(
            f"Eval seed collision for {path.stem}: EVAL_SEED={eval_base_seed} equals "
            f"train SEED={train_seed}. Set EVAL_SEED to a distinct value."
        )

    skills_path = raw.get("PAPER_EVAL_SKILLS_CKPT")
    meta_path = raw.get("PAPER_EVAL_META_CKPT")
    flat_path = raw.get("PAPER_EVAL_FLAT_PQN_CKPT")
    object_centric = bool(raw.get("PAPER_EVAL_OBJECT_CENTRIC", raw.get("OBJECT_CENTRIC", True)))

    # Resolve default skill ckpt from architecture SAVE_PATH when needed.
    needs_skills = kind in {"skills", "transfer", "hierarchical"}
    if needs_skills and not skills_path:
        skills_path = str(
            _skill_ckpt_from_save(
                cfg.get("SAVE_PATH", raw.get("SAVE_PATH", ".")),
                env_name if kind != "transfer" else "seaquest",
                train_seed,
            )
        )
        # Transfer always loads Seaquest skills
        if kind == "transfer":
            skills_path = raw.get("PAPER_EVAL_SKILLS_CKPT") or skills_path

    agent_kind = {
        "heuristic_random": "heuristic_random",
        "flat_pqn": "flat_pqn",
        "transfer": "transfer",
        "skills": "skills",
        "hierarchical": "skills",
        "tandem": "skills",
        "meta": "skills",
        "fixed_skills": "fixed_skills",
        "fixed_ego_rules": "fixed_ego_rules",
        "constant_ego": "constant_ego",
    }.get(kind, kind)

    # Allow transfer / fixed_skills YAMLs to select the baseline arm via mode.
    mode = str(cfg.get("SKILL_POLICY_MODE", raw.get("SKILL_POLICY_MODE", "learned"))).lower()
    if kind == "transfer" and mode in {"fixed_ego_rules", "constant_ego", "fixed_skills"}:
        agent_kind = "fixed_ego_rules" if mode != "constant_ego" else "constant_ego"
        skills_path = None
    elif kind == "fixed_skills":
        agent_kind = "constant_ego" if mode == "constant_ego" else "fixed_ego_rules"
        skills_path = None

    out_json = (
        PROJECT_ROOT
        / "outputs"
        / "models"
        / "paper_eval"
        / "eval"
        / f"{path.stem}_mods.json"
    )
    print(f"\n{'=' * 70}\n  EVAL mods: {path.relative_to(CONFIG_DIR)} kind={kind}\n{'=' * 70}")
    print(
        f"  train_SEED={train_seed}  EVAL_SEED={eval_base_seed}  "
        f"EVAL_NUM_SEEDS={eval_num_seeds}  episodes/seed={eval_episodes}",
        flush=True,
    )
    cfg_env = str(cfg.get("ENV_NAME", "")).lower().strip()
    raw_env = str(env_name).lower().strip()
    if kind in {
        "transfer",
        "fixed_skills",
        "fixed_ego_rules",
        "constant_ego",
    } and cfg_env and raw_env and cfg_env != raw_env:
        raise SystemExit(
            f"Transfer eval ENV mismatch: YAML ENV_NAME={raw_env!r} but architecture "
            f"config has ENV_NAME={cfg_env!r}. Add ENV_NAME to _EVAL_OVERRIDE_KEYS "
            f"(or drop PAPER_EVAL_ARCHITECTURE_OF). Refusing to run mislabeled eval."
        )
    cfg["ENV_NAME"] = env_name
    _pin_jaxatari_action_sets_to_host()
    try:
        run_mod_eval(
            agent_kind=agent_kind,
            config=cfg,
            env_name=env_name,
            skills_path=skills_path,
            meta_path=meta_path,
            flat_pqn_path=flat_path,
            object_centric=object_centric,
            trial=trial,
            mod_keys=list(raw["PAPER_EVAL_MOD_KEYS"]) if raw.get("PAPER_EVAL_MOD_KEYS") else None,
            mods_allowlist=str(raw.get("PAPER_EVAL_MODS") or env_name),
            num_episodes=eval_episodes,
            num_seeds=eval_num_seeds,
            base_seed=eval_base_seed,
            max_frames=eval_max_frames,
            output_json=out_json,
            extra_meta={
                "run_name": str(raw.get("WANDB_RUN_NAME", path.stem)),
                "config_stem": path.stem,
                "paper_eval_rq": raw.get("PAPER_EVAL_RQ", ""),
                "paper_eval_phase": raw.get("PAPER_EVAL_PHASE", "eval"),
                "paper_eval_kind": kind,
                "heuristic_mode": cfg.get("HEURISTIC_MODE"),
                "no_target_game_action": cfg.get("NO_TARGET_GAME_ACTION"),
                "train_seed": train_seed,
                "eval_seed": eval_base_seed,
                "eval_num_seeds": eval_num_seeds,
            },
            wandb_mode=wandb_mode,
        )
        # Optional Gallici-style greedy-8 (flat PQN only) → JSON + wandb summary.
        if kind == "flat_pqn" and bool(raw.get("PAPER_EVAL_PAPER_GREEDY8", False)):
            _run_paper_greedy8_after_mod_eval(
                raw,
                path=path,
                flat_path=flat_path,
                train_seed=train_seed,
                wandb_mode=wandb_mode,
            )
    finally:
        _release(label=f"eval:{path.stem}")


def _run_paper_greedy8_after_mod_eval(
    raw: dict[str, Any],
    *,
    path: Path,
    flat_path: str | None,
    train_seed: int,
    wandb_mode: str | None,
) -> None:
    """Run paper greedy-8 after CSV mod eval; never fail the suite on wandb I/O."""
    from concept_based_learning.paper_eval.paper_greedy8 import (
        DEFAULT_PAPER_GREEDY_MAX_FRAMES,
        DEFAULT_PAPER_GREEDY_SEEDS,
        publish_paper_greedy8_to_wandb,
        run_paper_greedy8_eval,
    )

    if not flat_path:
        print("  SKIP paper greedy-8: no PAPER_EVAL_FLAT_PQN_CKPT", flush=True)
        return
    ckpt = Path(str(flat_path))
    if not ckpt.is_file():
        # Allow relative paths from project root.
        alt = PROJECT_ROOT / flat_path
        ckpt = alt if alt.is_file() else ckpt
    if not ckpt.is_file():
        print(f"  SKIP paper greedy-8: missing ckpt {flat_path}", flush=True)
        return

    seeds_raw = raw.get("PAPER_EVAL_PAPER_GREEDY8_SEEDS")
    if isinstance(seeds_raw, (list, tuple)) and seeds_raw:
        eval_seeds = tuple(int(s) for s in seeds_raw)
    else:
        eval_seeds = DEFAULT_PAPER_GREEDY_SEEDS
    max_frames = int(
        raw.get("PAPER_EVAL_PAPER_GREEDY8_MAX_FRAMES", DEFAULT_PAPER_GREEDY_MAX_FRAMES)
    )
    out_json = (
        PROJECT_ROOT
        / "outputs"
        / "models"
        / "paper_eval"
        / "eval"
        / f"{path.stem}_paper_greedy8.json"
    )
    print(
        f"\n{'=' * 70}\n  PAPER greedy-8: {path.stem} "
        f"(seeds={list(eval_seeds)}, max_frames={max_frames})\n{'=' * 70}",
        flush=True,
    )
    summary = run_paper_greedy8_eval(
        ckpt,
        eval_seeds=eval_seeds,
        max_frames=max_frames,
        env_name=str(raw.get("ENV_NAME", "seaquest")),
        progress=True,
        output_json=out_json,
    )
    try:
        publish_paper_greedy8_to_wandb(
            summary,
            ckpt_dir=ckpt.parent,
            run_name=str(raw.get("WANDB_RUN_NAME", path.stem)),
            wandb_mode=wandb_mode,
            tags=list(raw.get("WANDB_TAGS") or []),
            entity=str(raw.get("ENTITY", "paul-seitz-tu-darmstadt")),
            project=str(raw.get("PROJECT", "concept_based_learning")),
        )
        print(
            f"  paper greedy-8 wandb: mean={summary['mean']:.1f} "
            f"(train_seed={train_seed})",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: paper greedy-8 wandb publish failed: {exc}", flush=True)


def _run_experiment(
    path: Path,
    *,
    wandb_mode: str | None,
    trial_steps: int | None,
    record_video: bool,
    seed: int | None,
    trial: bool,
    eval_episodes: int,
    eval_max_frames: int,
    enforce_rq_gate: bool,
    post_eval: bool,
    batch_tag: str | None = None,
    save_training_metrics_csv: bool = True,
) -> None:
    raw = _load_yaml(path)
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    status = str(raw.get("PAPER_EVAL_STATUS", "ready")).lower()
    if status in {"stub", "planned"}:
        print(f"\n  SKIP (status={status}): {path.stem}")
        return

    _bind_rtpt_job(path)

    if enforce_rq_gate and (
        raw.get("PAPER_EVAL_REQUIRES_RQ1")
        or kind in {"meta", "tandem", "transfer"}
        or (int(raw.get("PAPER_EVAL_RQ", 1)) >= 2 and phase == "finetune")
    ):
        _check_rq_gate([path], seed=int(seed if seed is not None else DEFAULT_SEED))

    # --video only records end-of-train base episodes for from-scratch trains
    # (eval / ZS / finetune / transfer stay silent to keep wandb + disk light).
    train_video = bool(record_video) and _is_from_scratch_train(raw, stem=path.stem)
    if record_video and not train_video:
        print(
            f"  video: skip (not from-scratch train): {path.stem}",
            flush=True,
        )

    train_kwargs = dict(
        wandb_mode=wandb_mode,
        trial_steps=trial_steps,
        record_video=train_video,
        seed=seed,
        batch_tag=batch_tag,
        save_training_metrics_csv=save_training_metrics_csv,
    )

    if phase == "eval" or kind in {
        "heuristic_random",
        "fixed_skills",
        "fixed_ego_rules",
        "constant_ego",
    }:
        _run_mod_eval(
            path,
            trial=trial,
            seed=seed,
            eval_episodes=eval_episodes,
            eval_max_frames=eval_max_frames,
            wandb_mode=wandb_mode,
        )
        return

    if kind == "flat_pqn":
        _run_flat_pqn_train(path, **train_kwargs)
        return

    if kind in {"skills", "meta", "tandem"}:
        # meta_only / tandem / skills_only all go through main.single_run
        _run_skills_train(path, **train_kwargs)
        # Optional frozen mod eval after train (weights unchanged — not finetune-on-mods).
        # Prefer explicit ``--phase eval`` for the paper protocol; use ``--post-eval``
        # only when you want train+eval in one shot (e.g. --smoke).
        if (
            post_eval
            and phase in {"finetune", "train"}
            and raw.get("PAPER_EVAL_MODS")
            and kind != "meta"
        ):
            from concept_based_learning.paper_eval.eval_harness import run_mod_eval

            cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
            env_name = str(cfg.get("ENV_NAME", "seaquest"))
            seed_i = int(cfg.get("SEED", seed or DEFAULT_SEED))
            save_root = Path(cfg.get("SAVE_PATH", "."))
            if not save_root.is_absolute():
                save_root = PROJECT_ROOT / save_root
            env_dir = save_root / env_name
            # skills_only: hierarchical_pqn_{env}_seed{N}.safetensors
            # tandem:      hierarchical_pqn_{env}_seed{N}_skills.safetensors
            candidates = [
                env_dir / f"hierarchical_pqn_{env_name}_seed{seed_i}.safetensors",
                env_dir / f"hierarchical_pqn_{env_name}_seed{seed_i}_skills.safetensors",
            ]
            ckpt = next((p for p in candidates if p.is_file()), None)
            meta_ckpt = env_dir / f"hierarchical_pqn_{env_name}_seed{seed_i}_meta.safetensors"
            if ckpt is not None:
                print(f"\n  Post-train mod eval for {path.stem} → {ckpt.name}")
                try:
                    run_mod_eval(
                        agent_kind="skills",
                        config=cfg,
                        env_name=env_name,
                        skills_path=str(ckpt),
                        meta_path=str(meta_ckpt) if meta_ckpt.is_file() else None,
                        trial=trial,
                        num_episodes=eval_episodes,
                        num_seeds=1,
                        base_seed=seed_i,
                        max_frames=eval_max_frames,
                        output_json=PROJECT_ROOT
                        / "outputs"
                        / "models"
                        / "paper_eval"
                        / "eval"
                        / f"{path.stem}_post_mods.json",
                        extra_meta={
                            "run_name": str(raw.get("WANDB_RUN_NAME", path.stem)),
                            "config_stem": path.stem,
                            "paper_eval_rq": raw.get("PAPER_EVAL_RQ", ""),
                            "paper_eval_phase": f"{phase}+post_eval",
                            "paper_eval_kind": kind,
                        },
                        wandb_mode=wandb_mode,
                    )
                finally:
                    _release(label=f"post_eval:{path.stem}")
            else:
                print(f"  WARN: expected checkpoint missing after train under {env_dir}")
        return

    if kind == "transfer":
        # Transfer configs are phase=eval; handled above. Fallback:
        _run_mod_eval(
            path,
            trial=trial,
            seed=seed,
            eval_episodes=eval_episodes,
            eval_max_frames=eval_max_frames,
            wandb_mode=wandb_mode,
        )
        return

    raise SystemExit(f"Unhandled PAPER_EVAL_KIND={kind!r} phase={phase!r} in {path}")


def _resolve_wandb_mode(*, trial: bool, wandb_mode: str | None) -> str:
    if wandb_mode is not None:
        return wandb_mode
    return "disabled" if trial else "online"


def _resolve_jaxatari_root(cli_value: str | None) -> Path | None:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("PAPER_EVAL_JAXATARI_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if DEFAULT_JAXATARI_FORK.is_dir():
        return DEFAULT_JAXATARI_FORK
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rq", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--phase", choices=["train", "eval", "finetune"], default=None)
    parser.add_argument("--only", nargs="*", metavar="NAME")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITE_STEMS),
        default=None,
        help=(
            "Curated multi-phase run list (ignores --phase). "
            f"{SKILL_RESHAPE_SUITE}: ego Seaquest train + mod eval "
            "+ Pong/BeamRider transfer (default --eval-max-frames "
            f"{SUITE_EVAL_MAX_FRAMES}). "
            f"{DESTROY_ABLATION_SUITE}: 3 DESTROY reward arms "
            "(y-align / standoff / contact-fail), lean 5-mod Seaquest + Pong base; "
            "use --gpus 0,1,2 so each GPU owns one arm. "
            f"{DESTROY_URGENCY_SUITE}: contact-fail + living tax (no shot ego); "
            "lean 5-mod Seaquest + Pong base. "
            f"{COLLECT_SHOT_PROBES_SUITE}: COLLECT living tax (NAVIGATE still "
            "shaped) and DESTROY shot-between bit; 2 GPUs, one arm each. "
            f"{EGO_POSE_PROBES_SUITE}: pose-only 12-dim vs DESTROY-only extras "
            "(COLLECT/AVOID/NAVIGATE pose, DESTROY pose+LoS+weapon); 2 GPUs. "
            f"{COLLECT_COMBAT_PROBES_SUITE}: urg-v2 clones — COLLECT death "
            "penalty off vs DESTROY-first (COLLECT never fights); 2 GPUs. "
            f"{NO_SURFACE_DEATH_SUITE}: urg-v2 clone trained with "
            "no_surfacing_death (0-diver surface is not a life loss). "
            f"{EGO_CLIP_BINS_SUITE}: urg-v2 + new heuristic; 2×2 of "
            "distance clip 50px and 8 overlapping world-θ bins "
            "(replace sin/cos(ψ)); 4 GPUs, one arm each. "
            f"{CLIP_BINS_NODECAY_SUITE}: retraining clip50+θ-bins with constant "
            "SKILL_LR, then full Seaquest/Pong/Tennis/BeamRider eval. "
            f"{CLIP50_ANGBINS_FULL_SUITE}: existing clip50+θ-bins ckpt; full "
            "Seaquest/Pong/Tennis/BeamRider tables, then RQ3 OC tandem "
            "(1 GPU sequential). "
            f"{THESIS_TIER_A_SUITE} (alias {THESIS_FINAL_SUITE}): ground-up Tier A — "
            "legacy clip50_angbins ego (pre-facing). Prefer "
            f"{THESIS_FACING_SUITE} for the overnight facing φ8 clip50 spine "
            "(COLLECT-surface ego + pixel/oc POV-mapper retrains + RQ2 zero-shot). "
            "RQ1 all arms + full SQ eval + RQ2 ego transfer (6 envs) + RQ3 pixel "
            "tandem (~2–3 GPUs overnight with --wait-deps). "
            f"{THESIS_TIER_B_SUITE}: RQ2 finetune 5M (legacy angbins; use --post-eval). "
            f"{THESIS_TIER_B_FACING_SUITE}: RQ2 finetune 5M from thesis_facing ckpts "
            "(collectsurface / pixel_pov / oc_pov; use --post-eval). "
            f"{THESIS_TIER_C_SUITE}: RQ2 from-scratch 25M + RQ3 meta/OC tandem "
            "(legacy angbins; use --post-eval). "
            f"{THESIS_TIER_C_FACING_SUITE}: facing POV spine — scratch ego/pixel, "
            "RQ3 meta/tandem from collectsurface, tandem-scratch BR/FR/AS "
            "(use --post-eval). "
            f"{NAVIGATE_SURFACE_TANDEM_PIXEL_SUITE}: 2-GPU — NAVIGATE-surface "
            "φ8 spine (+ pixel tandem). "
            f"{NAVIGATE_SURFACE_SURFACEGAP_SUITE}: 1-GPU — NAVIGATE-surface retrain "
            "with AABB surface-gap COLLECT/NAVIGATE/AVOID shaping (obs–reward aligned) "
            "→ SQ full eval → Pong ZS. "
            f"{KANGAROO_SURFACEGAP_SUITE}: Kangaroo train only (clip50+θ8 angbins + "
            "surface-gap rewards; Seaquest-proven spine). "
            f"{KANGAROO_FACING_SURFACEGAP_SUITE}: Kangaroo train only — POV facing "
            "ψ8 clip50 surfacegap (mirrors navigate_surface_surfacegap; "
            "post-heuristic-fix retrain). "
            f"{SURFACEGAP_DUAL_SUITE}: 2-GPU — kangaroo_surfacegap on GPU0 + "
            "navigate_surface_surfacegap train/eval/Pong on GPU1 (--wait-deps). "
            f"{NAVIGATE_SURFACE_BODYGAPS_SUITE}: 2-GPU — kangaroo facing bodygaps "
            "on GPU0 + Seaquest navigatesurface_bodygaps → eval → Pong on GPU1 "
            "(EGO_AABB_GEOM=body_gaps; --wait-deps). "
            f"{EGO_MIN_CHANNELS_SUITE}: 2-GPU — five v2 channel-deletion arms "
            "(noblocking / ±facing_enemy / ±weapon); trains first (3+2) then "
            "SQ full eval + Pong/Beamrider/Phoenix ZS (--wait-deps). "
            f"{EGO_MIN_DESTROYEXTRAS_SUITE}: locked noblocking_nofacing spine "
            "retrained with USE_DESTROY_ONLY_EGO_EXTRAS → SQ eval + Pong/BR/PH. "
            f"{EGO_MIN_NOCLIP_SUITE}: locked channels with EGO_DISTANCE_CLIP_MAX_PX=null "
            "(one-off Pong timing probe). "
            f"{EGO_MIN_NOWEAPON_OVALAVOID_SUITE}: drop can_fire + oval AVOID "
            "interrupt + anytime surface-sub dodge → SQ eval + Pong/BR/PH. "
            f"{EGO_MIN_NOWEAPON_OVALAVOID_SEEDS_SUITE}: same recipe seeds 45/46 "
            "(YAML SEED; do not pass --seed). "
            f"{EGO_MIN_NOWEAPON_HEU_ABLATION_SUITE}: noweapon oval_only vs "
            "surfacesub5 heuristic factors → SQ eval + Pong/BR/PH. "
            f"{EGO_MIN_NOFACING_OVALAVOID_SUITE}: Plan B — can_fire on "
            "(noblocking_nofacing) + oval/surface-sub heuristic. "
            f"{EGO_MIN_NOWEAPON_SURFACESUB5_CLIPSWEEP_SUITE}: noweapon+surfacesub5 "
            "distance clip sweep 10/20/30/50/80/100/noclip → SQ + Pong/BR/PH "
            "(clip50 reuses existing; --resume). "
            f"{EGO_MIN_NOWEAPON_SURFACESUB5_DISTABL_SUITE}: surfacesub5 clip0 "
            "(distance zeroed) / distbins8 (no hard clip) / clip10+sticky "
            "→ SQ + Pong/BR/PH. "
            f"{EGO_MIN_NOWEAPON_SURFACESUB5_DISTBANDS_SUITE}: exclusive "
            "near/mid/far (12/40px), no sticky → SQ + Pong/BR/PH. "
            f"{EGO_MIN_NOWEAPON_SURFACESUB5_CLIP50_STICKY_SUITE}: clip50 + "
            "train sticky 0.25 → SQ + Pong/BR/PH. "
            f"{EGO_MIN_NOWEAPON_SURFACESUB5_CLIPRAND_SUITE}: train-time "
            "per-episode clip sample {{10,20,30,50,80}}; eval/ZS fixed clip50 "
            "→ SQ + Pong/BR/PH. "
            f"{EGO_MIN_SURFACESUB5_CLIP10_CANFIRE_SUITE}: clip10 surfacesub5 + "
            "can_fire only (facing_enemy off) → SQ + Pong/BR/PH. "
            f"{EGO_MIN_SURFACESUB5_SKILLMECH_RW_SUITE}: ungated skill-mechanics "
            "reward regression (clip10+clip50 lock + clip10 RELABEL_ALL) → "
            "SQ eval + Pong/BR/PH; fresh …_skillmech stems. "
            f"{NEWSEAQUEST_SPAWNFIX_SUITE}: retrain PQN + clip10 skillmech "
            "(OFF_POLICY=false) on spawn-fixed Seaquest, then full SQ mods "
            "(fresh SAVE_PATHs; --wait-deps --gpus 2,3). "
            f"{ALE_PQN_MATCH_SUITE}: ALE CleanRL PQN W&B f163gauq HPs "
            "(10M / 128×32×2441 / LR 2.5e-4 / seed=2 / native_downscale=false) "
            "on JAXAtari Seaquest flat PQN; "
            "--suite ale_pqn_match --gpus 0 --wait-deps --skip-rq-gate --lean. "
            f"{PQN_PAPERHPS_OLDSEAQUEST_SUITE}: paper-HP PQN on old_seaquest "
            "×3 seeds (ENV_NAME=old_seaquest dense re-arm; EVAL_SEED≠SEED). "
            f"{NEWSEAQUEST_CLIP10_OFFPOLICY_SUITE}: clip10 skillmech on "
            "spawn-fixed Seaquest OFF_POLICY=true ×3 seeds (EVAL_SEED≠SEED). "
            f"{FIXED_EGO_SKILL_BASELINE_SUITE}: same scaffolding as clip10 "
            "skillmech transfer, but fixed ego-angle rules / constant ego "
            "actions instead of learned Q-heads (SQ + Pong/BR/PH). "
            f"{EGO_MIN_SURFACESUB5_DISTPOSE_SUITE}: clip50 surfacesub5 distance "
            "block → Δxy / sign(Δxy) / absolute xy (+ φ8); 3 trains → "
            "--gpus 0,1,2. "
            f"{EGO_MIN_SURFACESUB5_NEARTIERS_SUITE}: clip10 continuous near + "
            "exclusive tiers [10,20,50,80,150] beyond → SQ + Pong/BR/PH. "
            f"{EGO_MIN_SURFACESUB5_CANFIRE_OVERNIGHT_SUITE}: can_fire re-intro "
            "on clip10/50 + neartiers + deltaxy(φ8) + deltaxy(φ sincos); "
            "trains first → --gpus 0,1,2. "
            f"{ALE_CLIP10_HEU_SMOKE_SUITE}: clip10 spine ~25M native skills "
            "(PH/NTG/TP/AS/Pong/Breakout/BR/SI/Freeway/Montezuma) + SQ→target ZS; "
            "SQ 10M (+ sparse C/D/A 10M); Δxy skillmech neg-control + Pong/BR/PH; "
            "pixel+ego tandem on SQ+scratch except Montezuma "
            "(home mod eval + ZS from SQ tandem skills); trains first → --gpus 0,1,2. "
            f"{SPARSE_CDA_CLEAN_SUITE} / _gpu0|_gpu1|_gpu2: sparse C/D/A clean "
            "story — SQ 25M+eval+ZS (gpu0), BR+MZ 10M (gpu1), SQ pixel tandem "
            "20M+eval (gpu2); use bash scripts/run_sparse_cda_clean.sh --gpus … "
            "--video (not --lean). "
            f"{SPARSE_CDA_CLIP_SWEEP_SUITE}: sparse C/D/A Seaquest skills "
            "clip50 then clip100 (train+eval each); one GPU. "
            f"{SPARSE_CDA_CLIP_TRANSFER_SUITE} / _gpu0|_gpu1: ZS transfers for "
            "sparse clip50 (gpu0) then clip100 (gpu1) across clean-story envs "
            "(needs train ckpts); parallel: two terminals with "
            "--suite sparse_cda_clip_transfer_gpu{{0,1}} --wait-deps "
            "--skip-rq-gate. "
            f"{SPARSE_CDA_CLIP_EXTREMES_SUITE} / _gpu0|_gpu1: overnight "
            "clip0 train/eval/ZS (gpu0) + noclip train/eval/ZS then "
            "pixel-meta tandem 20M from clip10 sparse skills (gpu1); "
            "two terminals with --suite sparse_cda_clip_extremes_gpu{{0,1}} "
            "--wait-deps --skip-rq-gate. "
            f"{NOCLIP_SPARSE_SMOKE_SUITE} / _gpu0|_gpu1|_gpu2: noclip+sparse 20M "
            "ego scratch + flat PQN + from-scratch tandem on bases "
            "(SQ/PH/NTG/AS) + transfers (AS/TP rotate-aim facing motor); "
            f"{NOCLIP_SPARSE_SMOKE_RERUN_SUITE}: single-GPU --resume fill for "
            "cascade gaps; "
            f"{NOCLIP_SPARSE_SMOKE_AS_TP_SCRATCH_SUITE}: AS+TP skills+tandem "
            "trains only (no evals) after rotate-aim remount; "
            f"{NOCLIP_SPARSE_SMOKE_AS_TP_TANDEM_SUITE}: AS+TP tandem train/eval "
            "catch-up; "
            "three terminals with "
            "--suite noclip_sparse_smoke_gpu{{0,1,2}} --wait-deps --skip-rq-gate. "
            f"{EPS_SANITY_SUITE}: Seaquest skills ~5M ε schedule sanity "
            "(A/B/C/D/E; constant SKILL_LR; home mod gate + Pong/BR base ZS); "
            "locked ε=B (0.2→0.01/0.2). "
            f"{HP_SWEEP_SKILLS_SUITE}: LR grid + ε-decay05 probe with locked ε B; "
            "python scripts/materialize_hp_sweep.py --stage skills then "
            "--suite hp_sweep_skills --gpus … --wait-deps --skip-rq-gate. "
            f"{HP_SWEEP_TANDEM_SUITE}: clean-restart tandem meta mix grid "
            "(ε / LR×ε / switch×ε; skill ε B + SKILL_LR=1e-3; CNN [32, 64], "
            "death=0, EGO 18); "
            "python scripts/materialize_hp_sweep.py --stage tandem then "
            "--suite hp_sweep_tandem_v2 --gpus … --wait-deps --skip-rq-gate. "
            f"{TANDEM_E01_SCALE_SUITE}: locked e01 HPs at 20M + ~100M "
            "(+ home mod eval); "
            "python scripts/materialize_hp_sweep.py --stage tandem_e01 then "
            "--suite tandem_e01_scale --gpus … --wait-deps --skip-rq-gate. "
            f"{TANDEM_E01_OC_SCALE_SUITE}: locked e01 HPs at 20M for three OC "
            "metas (oc_simple_rel_role / oc_simple_role / oc_embed) + locked "
            "home mod eval — farming / robustness vs skill-only; "
            "python scripts/materialize_hp_sweep.py --stage tandem_e01_oc then "
            "--suite tandem_e01_oc_scale --gpus … --wait-deps --skip-rq-gate. "
            f"{NEXUS_PARITY_SUITE}: sanity — our locked HPs at NEXUS outer loop "
            "(50M / 1024 envs / 128 steps → 381 updates); tandem pixel first "
            "(OOM-fast) then skills + home evals; "
            "python scripts/materialize_nexus_parity.py then "
            "--suite nexus_parity --gpus 0 --wait-deps --skip-rq-gate --lean. "
            f"{OPTIM_SCALE_SQ_SUITE}: PQN optimal-scaling 1:1 HPs (50M / 1024 / "
            "NUM_STEPS=32, LR=2.5e-4+decay, MAX_GRAD_NORM=10, ε 1→0.001) × "
            "Seaquest ego-skills + tandem-pixel × seeds 44/45/46; "
            "tandem-first (OOM-fast); "
            "python scripts/materialize_optim_scale.py then "
            "--suite optim_scale_sq --gpus … --wait-deps --skip-rq-gate --lean. "
            f"{OPTIM_SCALE_PROBE_SUITE}: skills-only seed=1 probes on the same "
            "outer — full 1:1 vs ε B vs SKILL_LR=5e-4+ε B (no LR decay); "
            "run before the full suite to see what breaks; "
            "--suite optim_scale_probe --gpus … --wait-deps --skip-rq-gate --lean. "
            f"{OPTIM_SCALE_SKILLS_RERUN_SUITE}: Seaquest εB skills × seeds "
            "44/45/46 (nearest in-band AVOID/DESTROY heuristic) + home evals; "
            "trains-first for --gpus sharding; "
            "python scripts/materialize_optim_scale.py then "
            "--suite optim_scale_skills_rerun --gpus … --wait-deps --skip-rq-gate "
            "--lean. "
            f"{OPTIM_SCALE_SKILLS_BASES_SUITE}: post-NOOP locked skills "
            "(ε B + LR 2.5e-4+linear) on phoenix / namethisgame / asteroids "
            "(seed=44) + home mod eval. "
            f"{OPTIM_SCALE_SKILLS_TRANSFERS_SUITE}: same locked recipe on remaining "
            "transfer scratch envs (spaceinvaders, beamrider, timepilot, pong, "
            "breakout, freeway) + home mod evals; "
            "python scripts/materialize_optim_scale.py then "
            "--suite optim_scale_skills_transfers --gpus … --wait-deps "
            "--skip-rq-gate --lean. "
            f"{OPTIM_SCALE_SKILLS_CLIP_FOILS_SUITE}: locked εB × clip10/clip50 on "
            "Seaquest + foil envs (phoenix, asteroids, SI, BR, TP, Pong) — "
            "distance-regime probe vs noclip lock; trains-first for --gpus; "
            "--suite optim_scale_skills_clip_foils --gpus 0,1 --wait-deps "
            "--skip-rq-gate --lean. "
            f"{OPTIM_SCALE_SKILLS_LIVING_URGENCY_SUITE}: same envs, noclip εB, "
            "but COLLECT_LIVING_PENALTY + DESTROY_LIVING_PENALTY on "
            "(COLLECT_SPARSE_ONLY off) — discourage idle/spam-until-kill; "
            "--suite optim_scale_skills_living_urgency --gpus 0,1,2 "
            "--wait-deps --skip-rq-gate --lean. "
            f"{OPTIM_SCALE_SKILLS_LR_SUITE}: short LR probe seed=1 (ε B fixed) — "
            "lr1e4_lin / lr2p5e4_const / lr5e4_lin. "
            f"{OPTIM_SCALE_SKILLS_OLD_EPS_SUITE}: single YAML-ε arm (seed=44). "
            f"{SKILLS_HEU_CANFIRE_AB_SUITE}: seed45 εB skills A/B — noweapon vs "
            "can_fire-only (facing_enemy off), new heu (O₂~10%, no-diver→DESTROY) "
            "+ fixed home mods; "
            "--suite skills_heu_canfire_ab --gpus 0 --wait-deps --skip-rq-gate "
            "--lean. "
            f"{SKILLS_HEU_V2_TRANSFER_AB_SUITE}: heu-v2 εB A/B (noweapon vs "
            "can_fire) + Seaquest paper_core + ZS transfer × 9 envs "
            "(SI/BR/TP/AS/Pong/Breakout/Freeway/NTG/Phoenix) paper_core mods; "
            "python scripts/materialize_skills_heu_v2_transfer.py then "
            "--suite skills_heu_v2_transfer_ab --gpus 0 --wait-deps "
            "--skip-rq-gate --lean. "
            f"{SKILLS_IB_EGO_VS_PIXEL_SUITE}: IB ablation — fresh heu-v2 50M "
            "ego vs pixel_embed + target outline (same HPs/heuristic; new "
            "SAVE_PATHs, not pre-spawn-fix optim_scale heu_v2 ckpts); "
            "home paper_core evals; "
            "python scripts/materialize_skills_ib_ego_vs_pixel.py then "
            "--suite skills_ib_ego_vs_pixel --gpus 0 --wait-deps "
            "--skip-rq-gate --lean. "
            f"{OPTIM_SCALE_PQN_OC_SUITE} / {OPTIM_SCALE_PQN_PIXEL_SUITE}: flat PQN "
            "OC vs pixel @ same 1:1 HPs × seeds 44/45/46 + full Seaquest mod eval; "
            "train→eval sequential per seed — run one suite per GPU; "
            "python scripts/materialize_optim_scale.py then "
            "CUDA_VISIBLE_DEVICES=0 … --suite optim_scale_pqn_oc and "
            "CUDA_VISIBLE_DEVICES=1 … --suite optim_scale_pqn_pixel. "
            f"{PQN_PAPER_T5_SUITE}: PQN paper Table 5 outer (128 envs / 32 steps / "
            "100M agent = 400M frames; LR 2.5e-4 no decay; "
            "USE_NATIVE_DOWNSCALING=false) × flat PQN pixel + "
            "matched skills εB × seeds 44/45/46 + home paper_core evals; "
            "PQN evals also run paper greedy-8 → wandb; "
            "Seaquest paper ref ≈15899; "
            "python scripts/materialize_pqn_paper_table5.py then "
            "--suite pqn_paper_t5 --gpus … --wait-deps --skip-rq-gate --lean. "
            f"{PQN_PAPER_T5_PQN_SUITE}: PQN pixel trains+evals only "
            "(native_downscale=false re-run; no skills). "
            f"{OPTIM_SCALE_PQN_PIXEL_ENVS_SUITE}: flat PQN pixel @ 1:1 HPs × "
            "seeds 44/45/46 on phoenix/namethisgame/asteroids + transfer envs "
            "(SI/BR/TP/Pong/Breakout/Freeway) + same paper_core mod evals as "
            "skills — matched skills-vs-PQN robustness; trains-first for "
            "--gpus sharding; "
            "--suite optim_scale_pqn_pixel_envs --gpus … --wait-deps "
            "--skip-rq-gate --lean. "
            f"{OPTIM_SCALE_PQN_OC_BUILDS_SUITE}: flat PQN on tandem OC feature "
            "builds (oc_simple_rel_role / oc_simple_role / oc_embed) × seed 44 "
            "+ home mod eval — overnight format probe; "
            "--suite optim_scale_pqn_oc_builds --gpus … --wait-deps "
            "--skip-rq-gate --lean. "
            f"{FINAL_10M_BASE_SKILLS_SUITE}: ours-only ego skills @ 10M on paper "
            "bases (seaquest, phoenix, namethisgame, asteroids) + home paper_core "
            "evals; Time Pilot is a transfer (not included); "
            "python scripts/materialize_final_10m.py then "
            "--suite final_10m_base_skills --gpus 0 --wait-deps --skip-rq-gate "
            "--lean. "
            f"{FINAL_10M_PQN_PIXEL_GPU0_SUITE} / {FINAL_10M_PQN_PIXEL_GPU1_SUITE}: "
            "flat PQN pixel @ 10M + home evals (SQ+PH / NTG+AS). "
            f"{FINAL_10M_OVERNIGHT_GPU0_SUITE}: nexus_parity then "
            f"{FINAL_10M_PQN_PIXEL_GPU0_SUITE}; "
            f"{FINAL_10M_OVERNIGHT_GPU1_SUITE}: final_10m_base_skills then "
            f"{FINAL_10M_PQN_PIXEL_GPU1_SUITE} — two-GPU overnight. "
            f"{HP_SWEEP_TANDEM_CKPT16_SUITE}: archival eval-only for abandoned "
            "hp_sweep_tandem [16, 32] ckpts (not the overnight restart). "
            f"{SEAQUEST_META_OC_SPARSE_SUITE}: SQ tandem meta-OC input modes "
            "(oc_embed + oc_simple*) clip10 sparse C/D/A ~25M + home eval "
            "(vs pixel tandem sparse); --wait-deps --skip-rq-gate --video. "
            f"{DISTBINS8_PIPELINE_SUITE}: train distance-bins8 variant, then full "
            "Seaquest/Pong/BeamRider eval. "
            f"{KANGAROO_LIFETIME_SUITE}: Kangaroo noclip+angbins skills train "
            "(default global target-cut). "
            f"{CLIP50_LIFETIME_PIPELINE_SUITE}: Seaquest clip50+θ-bins train with "
            "SKILL_TARGET_CUT=lifetime, then full SQ mods + Pong full + Phoenix base. "
            f"{TIERSCALAR_PIPELINE_SUITE}: single-channel tier scalar for both "
            "distance and angle, then full Seaquest/Pong/BeamRider eval. "
            f"{ANGSCALAR_ONLY_PIPELINE_SUITE}: single-channel angle tiers only "
            "(no distance clip/bins), then full Seaquest/Pong/BeamRider eval. "
            f"{SCALAR_ABLATION_DUAL_SUITE}: 2-GPU paired run of tierscalar and "
            "angscalar-only arms (train + eval + Pong + BeamRider each). "
            f"{MINIMAL_ABLATION_SUITE}: 6 isolated simplification ablations "
            "(32-sector angle encodings, destroy reward flags, no-LoS) with "
            "full Seaquest eval + Pong/BeamRider base transfer; 4 GPUs "
            "(train-first stem order, then --resume after reboot). "
            f"{MINIMAL_ABLATION_TRAIN_SUITE}: the six RQ1 trains only (~3h on "
            "3 GPUs); run {MINIMAL_ABLATION_SUITE} --eval-only --resume next day. "
            f"{CLIP_LOW_PIPELINE_SUITE}: clip15 vs clip8 distance (same 8 θ-bins) "
            "+ locked clip50 re-eval with legal-ego decode; "
            "full Seaquest + Pong/BeamRider/Phoenix base; 2 GPUs. "
            f"{CLIP_SWEEP_PIPELINE_SUITE}: distance clip sweep "
            "(0/30/60/80/100/120/noclip trains + dual-mask Pong/Beam/Phoenix; "
            "reuses clip8/15/50 ckpts; SQ full once per new arm); 2 GPUs overnight. "
            f"{DISTANCE_ENCODE_PIPELINE_SUITE}: fixed distance tiers/flags "
            "(tier2/4/6, flags3/6) + 8 θ-bins; maskoff; SQ full + transfers. "
            f"{PIXEL_NOOUTLINE_SUITE}: eval-only — Seaquest pixel ckpt with "
            "target outline OFF (RQ1 mods + 6 RQ2 zero-shot transfers). "
            f"{PQN_TRANSFER_SUITE}: eval-only — Seaquest flat PQN zero-shot on "
            "6 RQ2 envs (control; transfer should collapse)."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "With --suite: skip train configs (requires existing ckpts). "
            "Valid for skill_reshape, destroy_ablation, destroy_urgency, "
            "collect_shot_probes, ego_pose_probes, collect_combat_probes, "
            "no_surface_death, ego_clip_bins, clip_bins_nodecay, and "
            "clip50_angbins_full, thesis_tier_a, thesis_tier_b, thesis_tier_c, "
            "thesis_final, distbins8_pipeline, kangaroo_lifetime, "
            "clip50_lifetime_pipeline, tierscalar_pipeline, "
            "angscalar_only_pipeline, scalar_ablation_dual, minimal_ablation, "
            "minimal_ablation_train, clip_low_pipeline, clip_sweep_pipeline, "
            "and distance_encode_pipeline "
            "(train-only suites: use companion --eval-only after)."
        ),
    )
    parser.add_argument("--shard", metavar="INDEX/TOTAL")
    parser.add_argument(
        "--gpus",
        default=None,
        metavar="IDS",
        help=(
            "Comma-separated GPU ids for a one-shot multi-GPU launch "
            "(e.g. 0,1). Spawns one worker per GPU with --shard i/N --wait-deps "
            "and blocks until all finish."
        ),
    )
    parser.add_argument(
        "--wait-deps",
        action="store_true",
        help=(
            "If RQ1 ckpts required by a run are missing, poll until they appear "
            "instead of failing the RQ gate (for multi-GPU overnight)."
        ),
    )
    parser.add_argument(
        "--wait-deps-poll",
        type=float,
        default=60.0,
        help="Seconds between dependency polls when --wait-deps is set (default: 60).",
    )
    parser.add_argument(
        "--wait-deps-timeout-h",
        type=float,
        default=36.0,
        help="Hours to wait for deps before failing (default: 36).",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help=(
            f"Abort immediately if free space on the project volume drops below "
            f"this many GiB (default: {DEFAULT_MIN_FREE_GB}). Checked at suite "
            "start, before each run, and during --wait-deps polls. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--batch-tag",
        default=None,
        help=(
            "W&B batch tag (default: paper_eval_YYYYMMDD_HHMM). "
            "Also set via PAPER_EVAL_BATCH_TAG. Added as batch:<tag> alongside suite:paper_eval."
        ),
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--list-mods", nargs="?", const="*", metavar="ENV")
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--trial-steps", type=int, default=TRIAL_STEPS_DEFAULT)
    parser.add_argument("--smoke", action="store_true", help="Run curated wiring smoke matrix")
    parser.add_argument(
        "--post-eval",
        action="store_true",
        help=(
            "After train/finetune, also run frozen mod eval on the new checkpoint. "
            "Default off (except --smoke). Prefer --phase eval for the paper protocol."
        ),
    )
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--video",
        action="store_true",
        help=(
            "Record end-of-train wandb videos for from-scratch base trains only "
            "(default: off; skips eval/ZS/finetune/transfer)."
        ),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Deprecated alias: videos are off by default.",
    )
    parser.add_argument(
        "--lean",
        action="store_true",
        help=(
            "Minimize local disk writes: no videos, no training_metrics.csv, "
            "wandb local dirs under $TMPDIR/$USER/cbl_paper_eval."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip experiments that already have their checkpoint OR a complete "
            "mod-eval JSON on disk (all allowlisted mods present, including base). "
            "Missing/partial evals from disk-full failures are re-queued. "
            "Filter runs before sharding so GPUs only share remaining work."
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="STEM",
        help=(
            "Force-skip these config stems and any run that requires them as an RQ1 dep "
            "(e.g. --exclude seaquest_skills_pixel also drops pixel transfer/finetune)."
        ),
    )
    parser.add_argument("--abort-on-failure", action="store_true")
    parser.add_argument("--skip-rq-gate", action="store_true")
    parser.add_argument("--jaxatari-root", default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument(
        "--eval-max-frames",
        type=int,
        default=None,
        help=(
            f"Per-episode frame cap for mod/transfer evals "
            f"(default: {SUITE_EVAL_MAX_FRAMES}; trial/smoke: 400)."
        ),
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Regenerate RQ2/RQ3 YAMLs then exit",
    )
    args = parser.parse_args()

    if args.materialize:
        import importlib.util

        mat_path = PROJECT_ROOT / "scripts" / "materialize_paper_eval_configs.py"
        spec = importlib.util.spec_from_file_location("materialize_paper_eval_configs", mat_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.main()
        return

    if args.gpus:
        gpu_ids = [g.strip() for g in str(args.gpus).split(",") if g.strip() != ""]
        if len(gpu_ids) < 1:
            raise SystemExit("--gpus needs at least one id (e.g. --gpus 0,1)")
        if args.shard:
            raise SystemExit("Use either --gpus or --shard, not both ( --gpus adds --shard )")
        batch_tag = _resolve_batch_tag(args.batch_tag)
        os.environ["PAPER_EVAL_BATCH_TAG"] = batch_tag
        print(f"paper_eval batch tag: batch:{batch_tag}", flush=True)
        # Parent only orchestrates; workers set jaxatari / CUDA / --lean themselves.
        raise SystemExit(_launch_multi_gpu(sys.argv[1:], gpu_ids))

    if not CONFIG_DIR.is_dir():
        raise SystemExit(f"Missing {CONFIG_DIR}")

    if args.list_mods is not None:
        _print_mods(None if args.list_mods == "*" else args.list_mods)
        return

    _prepend_jaxatari_root(_resolve_jaxatari_root(args.jaxatari_root))

    only = list(args.only) if args.only else None
    suite = str(args.suite).strip() if args.suite else None
    phase_filter = args.phase
    if args.smoke:
        only = list(SMOKE_STEMS)
    elif suite:
        if args.only:
            raise SystemExit("Use either --suite or --only, not both")
        if args.eval_only:
            only = []
            for s in SUITE_STEMS[suite]:
                matches = list(CONFIG_DIR.rglob(f"{s}.yaml"))
                if not matches:
                    raise SystemExit(f"Config(s) not found: {[s]}")
                phase = str(_load_yaml(matches[0]).get("PAPER_EVAL_PHASE", "train")).lower()
                if phase != "train":
                    only.append(s)
        else:
            only = list(SUITE_STEMS[suite])
        # Suites span train+eval; do not filter by --phase.
        phase_filter = None
        print(f"paper_eval suite={suite} ({len(only)} run(s))", flush=True)
    elif args.eval_only:
        raise SystemExit("--eval-only requires --suite")

    paths = _discover_configs(rq=args.rq, only=only, phase=phase_filter)
    if not paths:
        raise SystemExit("No matching configs (run --materialize if RQ2/RQ3 missing)")

    # Preserve smoke / suite / --only order (discovery is alphabetical).
    if only:
        by_stem = {p.stem: p for p in paths}
        ordered = [by_stem[s] for s in only if s in by_stem]
        missing = [s for s in only if s not in by_stem]
        if missing:
            raise SystemExit(f"Config(s) not found: {missing}")
        paths = ordered
    else:
        # Overnight / multi-GPU: ungated RQ1 trains before gated eval/RQ2/RQ3.
        paths = sorted(paths, key=_schedule_key)

    seed = args.seed if args.seed is not None else DEFAULT_SEED
    exclude = {str(s).removesuffix(".yaml") for s in (args.exclude or [])}
    if args.resume or exclude:
        paths = _filter_remaining_paths(
            paths,
            seed=seed,
            exclude=exclude,
            skip_done=bool(args.resume),
        )
        if not paths:
            raise SystemExit("Nothing left to run (all done or excluded).")

    if args.shard:
        idx, n = _parse_shard(args.shard)
        paths = _shard_paths(paths, shard_idx=idx, num_shards=n)
        if not paths:
            raise SystemExit(f"Shard {args.shard} empty")

    if args.list:
        _print_matrix(paths)
        if args.resume or exclude:
            print("\n(status with --resume / --exclude applied above)")
        return

    # Smoke / skill-reshape train first in-order; gate per-run instead of upfront.
    # --wait-deps: skip the upfront fail; each run polls instead.
    suite_will_train = bool(suite and not args.eval_only)
    skip_gate = bool(
        args.skip_rq_gate or args.smoke or args.wait_deps or suite_will_train
    )
    if not skip_gate:
        gated = [
            p
            for p in paths
            if _load_yaml(p).get("PAPER_EVAL_REQUIRES_RQ1")
            or (
                int(_load_yaml(p).get("PAPER_EVAL_RQ", 1)) >= 2
                and str(_load_yaml(p).get("PAPER_EVAL_PHASE")) != "train"
            )
            or str(_load_yaml(p).get("PAPER_EVAL_KIND", "")).lower() in {"meta", "tandem"}
        ]
        if gated:
            _check_rq_gate(gated, seed=seed)

    wandb_mode = _resolve_wandb_mode(trial=args.trial or args.smoke, wandb_mode=args.wandb_mode)
    os.environ["PAPER_EVAL_WANDB_MODE"] = str(wandb_mode)
    trial_steps = args.trial_steps if (args.trial or args.smoke) else None
    trial = bool(args.trial or args.smoke)
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else (1 if trial else 3)
    if args.eval_max_frames is not None:
        eval_max_frames = int(args.eval_max_frames)
    elif trial:
        eval_max_frames = 400
    else:
        # Default full-game horizon for every suite / --only eval (was 800 unless
        # the suite was on an allowlist — thesis_tier_* accidentally stayed short).
        eval_max_frames = SUITE_EVAL_MAX_FRAMES
    batch_tag = _resolve_batch_tag(args.batch_tag)
    os.environ["PAPER_EVAL_BATCH_TAG"] = batch_tag
    print(f"paper_eval batch tag: batch:{batch_tag}")
    print(
        f"paper_eval eval_episodes={eval_episodes} eval_max_frames={eval_max_frames}",
        flush=True,
    )

    lean = bool(args.lean)
    if lean:
        _configure_lean_disk()

    # Videos off by default; --video opts in. --lean always disables videos.
    record_video = bool(args.video) and not lean and not trial and not args.no_video
    save_training_metrics_csv = not lean

    min_free_gb = float(args.min_free_gb)
    _assert_disk_space(min_free_gb=min_free_gb, where="suite start")
    print(
        f"disk: {_free_gb():.2f} GiB free (abort below {min_free_gb:.2f} GiB); "
        f"videos={'on' if record_video else 'off'}; "
        f"training_csv={'on' if save_training_metrics_csv else 'off'}"
    )

    print(f"paper_eval: {len(paths)} run(s)")
    for p in paths:
        print(f"  - {p.relative_to(CONFIG_DIR)}")
    print()

    _init_rtpt_for_paths(
        paths,
        experiment_name=_rtpt_name(args.shard, args.rq, suite=suite),
        trial_steps=trial_steps,
        trial=trial,
    )

    failures: list[tuple[str, str]] = []
    failed_stems: set[str] = set()
    for path in paths:
        try:
            _assert_disk_space(
                min_free_gb=min_free_gb,
                where=f"before {path.stem}",
            )
            if args.wait_deps:
                _wait_for_path_deps(
                    path,
                    seed=seed,
                    poll_s=float(args.wait_deps_poll),
                    timeout_h=float(args.wait_deps_timeout_h),
                    min_free_gb=min_free_gb,
                    failed_local_stems=failed_stems,
                )
            # Reclaim VRAM left by the previous JAX train/eval before compiling again.
            _release(label=f"before:{path.stem}")
            _run_experiment(
                path,
                wandb_mode=wandb_mode,
                trial_steps=trial_steps,
                record_video=record_video,
                seed=args.seed,
                trial=trial,
                eval_episodes=eval_episodes,
                eval_max_frames=eval_max_frames,
                enforce_rq_gate=bool(args.smoke) and not args.skip_rq_gate,
                post_eval=bool(args.post_eval or args.smoke),
                batch_tag=batch_tag,
                save_training_metrics_csv=save_training_metrics_csv,
            )
        except SystemExit:
            raise
        except Exception as exc:
            traceback.print_exc()
            failures.append((path.name, str(exc)))
            failed_stems.add(path.stem)
            if _is_no_space_error(exc):
                print(
                    f"\n  FATAL disk full during {path.name}; aborting suite "
                    f"(not waiting on further deps).\n",
                    flush=True,
                )
                break
            if args.abort_on_failure:
                break
            print(f"\n  FAILED (continuing): {path.name}: {exc}\n")

    if failures:
        print("\nFailed runs:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        raise SystemExit(1)
    print("\nAll paper_eval runs finished successfully.")


if __name__ == "__main__":
    main()
