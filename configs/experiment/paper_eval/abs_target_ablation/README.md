# Dedicated absolute target ablation

Ten 50M-step training configurations (Seaquest and Name This Game, seeds 44–48).
Regenerate with `python scripts/materialize_abs_target_ablation.py`. Each is derived
from the matching `thesis_final/tf_{game}_skills_ego_seed{seed}.yaml`; only skill
input mode, checkpoint destination, run name and arm/input tags differ.

`ego_abs_target` provides 4 frames × 9 channels, frame-major:
`[target_cx/160, target_cy/210, target_w/160, target_h/210, active,
player_cx/160, player_cy/210, player_w/160, player_h/210]`.
All nine channels are zero when that frame's target is inactive. No relative
features, other objects, labels or oxygen enter the skill network. Facing-relative
actions, heuristic selection, rewards, optimizer and MLP match SCOPE.

The shared mode registry dispatches the complete vector in training and evaluation.
Network initialization infers its width from a real probe observation: 36, without
an ego prefix. `ego_context_extra_dim` retains its additive-only contract; it is
not used to initialize this network. Equal-width SCOPE checkpoints cannot be
distinguished by shape; preserve the input-mode config sidecar with checkpoints.

## Validation (2026-11-11)

- New suite: 10 passed, covering both primary games, normalized geometry,
  inactive masking, target selection, restriction, translation sensitivity,
  bottom-edge y, JIT/vmap, MLP width and reference-config parity.
- Existing `tests/test_ego_context.py`: 20 passed.
- Thesis `figures/evaluation/test_data.py`: 6 passed.
- Synthetic end-to-end pipeline check: imported ten aggregated results into a
  temporary snapshot, generated factor tables and four figures, and verified
  tables/active RQ2 figures handle missing absolute-target results.
- CPU smoke: 100,352 transitions, safetensors save/reload with exact tensor
  equality, finite weights, 36×128 first layer, one shifted-lanes evaluation
  capped at 256 agent steps. Passed; return 0 is only a smoke result.
  Artifacts: `tmp/ego_abs_target_smoke_20261111/`, including `smoke_passed.json`.

Commands from the CBL repository:

```bash
PYTHONPATH=src:. JAX_PLATFORMS=cpu python -m pytest tests/test_ego_abs_target.py tests/test_ego_context.py -q
PYTHONPATH=src:. JAX_PLATFORMS=cpu WANDB_MODE=disabled python scripts/smoke_abs_target_ablation.py --output tmp/abs_target_smoke_fresh --steps 100352 --max-frames 256
```

The smoke output directory must be new. Smoke overrides use 16 environments,
8 steps, 4 minibatches and one epoch for CPU validation; production YAMLs retain
SCOPE's training settings.

## Production schedule awaiting planning

No full runs launched. Plan ten training runs followed by 70 checkpoint/mod
jobs: five seeds × seven conditions × two games, nine episodes per job (630 total).
Use the seven conditions in each game's thesis `data.MODS` list. No transfer jobs.
Check GPU processes before placement; keep evaluation artifacts distinct from
smoke artifacts. Aggregate each checkpoint's seven conditions under canonical
`tf_{game}_skills_ego_abstarget_seed{seed}_eval_mods.json` names.

In the thesis repository, the snapshot builder accepts `--abs-target-results DIR`
for those aggregated files. Select a fresh `--output` snapshot path. Existing
frozen snapshots and generated thesis assets have not been overwritten. The arm
key is `abs_target`, displayed as `+absolute target`; a thesis letter is deferred.
