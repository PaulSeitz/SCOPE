# Avsep collect / surface diagnosis — 2026-10-26

Frozen checkpoint only. No retraining until the three failure modes below are separated.

## Checkpoint

- `outputs/models/paper_eval/thesis_final/tf_seaquest_skills_ego_avsep_seed44/seaquest/hierarchical_pqn_seaquest_seed44.safetensors`
- Paper config: `configs/experiment/paper_eval/thesis_final/tf_seaquest_skills_ego_avsep_seed44_eval.yaml`
- Home eval (already finished): base 11489, no_enemies 14107 (high variance)

## Plausible event miss (not yet measured)

Seaquest does not stamp `was_collected`. COLLECT success uses rising AABB overlap, with a
fallback that requires the player to overlap the diver’s **previous** box after the
repeated action. Across four simulation frames a diver can move, be collected, and
disappear so endpoint geometry misses a genuine pickup.

## Episodes to instrument (`no_enemies`)

Mod-eval seed formula: `ep_seed = 10044 + seed_idx * 10000 + episode_idx * 997`

| role | seed_idx | ep | ep_seed | eval return | rescues |
|---|---:|---:|---:|---:|---:|
| low | 2 | 0 | 30044 | 1460 | 1 |
| mid | 0 | 0 | 10044 | 3700 | 2 |
| high | 1 | 0 | 20044 | 58400 | 11 |

## Logger

`scripts/diagnose_seaquest_collect_events.py` writes per-step JSONL plus a summary:

- selected target / skill, player xy, oxygen, carried divers
- env pickups (`collected_divers`↑) and rescues (surface clear of a full rack)
- COLLECT reward/terminal and reason (`was_collected`, `overlap_rising_live`,
  `collect_in_place`, `missed_deactivation`, …)
- time from reaching 6 divers → first surface-WP selection → rescue
- life losses with oxygen vs unsuccessful-surface heuristic

## Decision table

| Observation | Next action |
|---|---|
| Real pickups or waypoint arrivals without intended feedback | Repair event detection before changing rewards. |
| Six divers carried but waypoint not selected appropriately | Inspect the selector and its overrides. |
| Waypoint selected correctly but skill stalls / moves wrong | Investigate COLLECT execution and its training data. |

Do not change reward objectives while this is open. AVOID shaping / kill-mode /
contact fixes landed together; a COLLECT symptom may still come from shared trajectories.

## Live base play (for watching)

```bash
cd ~/ConceptBasedLearning && source /tmp/cbl_queue_env.sh
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled
$PY -m concept_based_learning.pqn_visualizer \
  --model outputs/models/paper_eval/thesis_final/tf_seaquest_skills_ego_avsep_seed44/seaquest/hierarchical_pqn_seaquest_seed44.safetensors \
  --paper-config configs/experiment/paper_eval/thesis_final/tf_seaquest_skills_ego_avsep_seed44_eval.yaml \
  --seed 10044 \
  --wandb-mode disabled \
  --display pygame
```

Omit `--game-mods` for base. Add `--game-mods no_enemies` for the empty sea.
