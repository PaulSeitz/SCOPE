# Seaquest interface probes — scheduler handoff

Prepared 2026-10-26. Three independent seed-44 development probes. No jobs launched.

| Run | Token | Earlier diver selection | Geometry |
| --- | --- | --- | --- |
| A | a_early | enabled when no visible DESTROY/AVOID object | centre_ray |
| B | b_closest | current reference selector | closest_points |
| C | c_combined | same as A | closest_points |

The completed repaired tf_seaquest_skills_ego_avsep_seed44 is the unchanged-selector, centre-ray reference. All new probes train from scratch. Do not initialize from its weights.

Earlier selection bypasses only the near-distance/underwater gate. Diver screen margins, full-capacity exclusion, danger interrupts and oxygen/surfacing priorities remain in force. A threat is an active DESTROY or AVOID hitbox intersecting the screen, including hostile projectiles and the surface submarine. Offscreen spawns do not block the new rule. The new COLLECT_EARLY_WHEN_NO_THREATS flag defaults to false.

In B/C, distance and facing-relative bearing use the same signed closest-point separation vector. Transfer retains this geometry. AVOID shaping uses the new distance too. The heuristic itself retains its existing centre-ray distance for ranking targets.

## Common settings

- Seaquest seed 44. Same requested 50M budget, 1024 environments, 32 steps, 2 epochs, 256 minibatches.
- Corrected lambda returns and half-MSE. LR 0.00025 with linear decay, gamma 0.99, lambda 0.65.
- Current repaired contact/event code. AVOID_SPARSE_ONLY=false, AVOID_ESCAPE_BONUS=0.01, AVOID_KILLED_EVENT_MODE=none.
- COLLECT_SPARSE_ONLY=true, COLLECT_DEATH_PRECEDENCE=death_first, SKILL_DEATH_PENALIZE_COLLECT=true, DESTROY_CONTACT_FAIL=true.
- Cross-skill updating on, alternative-target relabeling off.
- Four frames, eight bearing bins, no extra channels. Same 36 -> 128 -> 128 -> 128 -> 18 architecture and facing-relative actions.
- Home evaluation on all nine Seaquest conditions. Eval seeds 10044/10045/10046 with three episodes each, nine episodes per condition.
- Frozen transfers to base NTG, Space Invaders, Asteroids, Pong, Breakout, Freeway, Phoenix, Beamrider. Nine episodes per target. No target training. No Time Pilot.

## Deploy before scheduling

Copy the current working tree and all 30 generated probe YAMLs to the cluster. Include the repaired environments/skill_rewards.py and these changed files.

- src/concept_based_learning/environments/seaquest/heuristic_policy.py
- src/concept_based_learning/environments/heuristics.py
- src/concept_based_learning/paper_eval/config_merge.py
- src/concept_based_learning/paper_eval/eval_harness.py
- scripts/run_paper_eval.py
- scripts/materialize_seaquest_interface_probes.py
- configs/experiment/paper_eval/thesis_final/_seaquest_interface_probes_manifest.yaml

Keep the corrected trainer/returns code and Seaquest robustness fork. The checkpoint-sidecar overlay now retains source EGO_AABB_GEOM. Check interpreter and imported fork on the cluster.

Regenerate only these configs with python -B scripts/materialize_seaquest_interface_probes.py. Do not regenerate the full thesis suite for this diagnostic wave.

## Worker chains

Assign three free GPU IDs to GPU_A, GPU_B, GPU_C. Start these chains concurrently from the CBL root. Retain the scheduler's normal W&B mode.

```bash
export PYENV_VERSION=jaxtari-fork
export JAXATARI_ROOT=/home/paul/Documents/JAXAtari-seaquest-robustness
export PYTHONPATH="$JAXATARI_ROOT/src:$PWD/src"
```

Adjust the fork path for the cluster. Set the GPU variables explicitly. Do not use --trial or --post-eval. The home eval already appears explicitly.

### Run A

```bash
python -u scripts/run_paper_eval.py --only \
  tf_seaquest_skills_ego_probe_a_early_seed44 \
  tf_seaquest_skills_ego_probe_a_early_seed44_eval \
  tf_zs_skills_ego_probe_a_early_seaquest_to_namethisgame_seed44 \
  tf_zs_skills_ego_probe_a_early_seaquest_to_spaceinvaders_seed44 \
  tf_zs_skills_ego_probe_a_early_seaquest_to_asteroids_seed44 \
  tf_zs_skills_ego_probe_a_early_seaquest_to_pong_seed44 \
  tf_zs_skills_ego_probe_a_early_seaquest_to_breakout_seed44 \
  tf_zs_skills_ego_probe_a_early_seaquest_to_freeway_seed44 \
  tf_zs_skills_ego_probe_a_early_seaquest_to_phoenix_seed44 \
  tf_zs_skills_ego_probe_a_early_seaquest_to_beamrider_seed44 \
  --gpus "$GPU_A" --wait-deps --skip-rq-gate --lean --resume
```

### Run B

```bash
python -u scripts/run_paper_eval.py --only \
  tf_seaquest_skills_ego_probe_b_closest_seed44 \
  tf_seaquest_skills_ego_probe_b_closest_seed44_eval \
  tf_zs_skills_ego_probe_b_closest_seaquest_to_namethisgame_seed44 \
  tf_zs_skills_ego_probe_b_closest_seaquest_to_spaceinvaders_seed44 \
  tf_zs_skills_ego_probe_b_closest_seaquest_to_asteroids_seed44 \
  tf_zs_skills_ego_probe_b_closest_seaquest_to_pong_seed44 \
  tf_zs_skills_ego_probe_b_closest_seaquest_to_breakout_seed44 \
  tf_zs_skills_ego_probe_b_closest_seaquest_to_freeway_seed44 \
  tf_zs_skills_ego_probe_b_closest_seaquest_to_phoenix_seed44 \
  tf_zs_skills_ego_probe_b_closest_seaquest_to_beamrider_seed44 \
  --gpus "$GPU_B" --wait-deps --skip-rq-gate --lean --resume
```

### Run C

```bash
python -u scripts/run_paper_eval.py --only \
  tf_seaquest_skills_ego_probe_c_combined_seed44 \
  tf_seaquest_skills_ego_probe_c_combined_seed44_eval \
  tf_zs_skills_ego_probe_c_combined_seaquest_to_namethisgame_seed44 \
  tf_zs_skills_ego_probe_c_combined_seaquest_to_spaceinvaders_seed44 \
  tf_zs_skills_ego_probe_c_combined_seaquest_to_asteroids_seed44 \
  tf_zs_skills_ego_probe_c_combined_seaquest_to_pong_seed44 \
  tf_zs_skills_ego_probe_c_combined_seaquest_to_breakout_seed44 \
  tf_zs_skills_ego_probe_c_combined_seaquest_to_freeway_seed44 \
  tf_zs_skills_ego_probe_c_combined_seaquest_to_phoenix_seed44 \
  tf_zs_skills_ego_probe_c_combined_seaquest_to_beamrider_seed44 \
  --gpus "$GPU_C" --wait-deps --skip-rq-gate --lean --resume
```

Each chain runs one training, one home eval, then eight frozen transfers. Total is three training runs and 27 evaluation runs. All output paths include the probe token.

Retain negative outcomes and inspect the first dive and complete collection/surfacing cycles alongside per-episode scores. These seed-44 probes are not a replacement final suite. Replicate a chosen recipe before expanding it.
