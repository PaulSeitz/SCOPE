# NEXUS integration (symbolic_options)

Upstream: https://github.com/remunds/symbolic_options (`develop`)

## Status (normal OC modus operandi)

**Default:** use `JAXATARI_ROOT` / the installed robustness Seaquest fork (what the
cluster already has). `--vendored-seaquest` is opt-in for NEXUS’s bundled old copy.

**Usable:**

- Submodule + `NEXUS_ROOT` / `JAXATARI_ROOT` path wiring
- Alg overlays (paper NEXUS symbolic, our HPs, PQN, HRL/HPQN)
- Train via `scripts/run_nexus.py train …` (patches `reward_funcs` onto the
  installed `JaxSeaquest` so NEXUS skill rewards work)
- **Full paper_eval mod allowlist** via `mod-eval` → `jaxatari.make(..., mods=…)`
  (same API as RQ1), writing `*_mods.json` / CSV under `outputs/models/paper_eval/eval/`

**Theoretical ToDo (not for this OC path):**

- Pixel-input HPQN/HRL baseline (shared CNN for meta + skills) for fair
  comparison to our pixel PQN; full pixel NEXUS (symbolic/nesy) is out of scope

## Cluster / path setup

The repo is a **git submodule** at `third_party/symbolic_options`.

```bash
git submodule update --init --recursive third_party/symbolic_options

# Point at any checkout (local or cluster scratch):
export NEXUS_ROOT=/path/to/symbolic_options          # default: $PWD/third_party/symbolic_options
export JAXATARI_ROOT=/path/to/JAXAtari-seaquest-robustness
# alias also accepted:
export PAPER_EVAL_JAXATARI_ROOT=$JAXATARI_ROOT

cd "$NEXUS_ROOT" && uv sync   # or pip install -e .
```

`scripts/run_nexus.py` syncs CBL alg overlays into `$NEXUS_ROOT/.../config/alg/`,
puts **`$JAXATARI_ROOT/src` first on `PYTHONPATH`** (installed Seaquest), patches
NEXUS `reward_funcs` onto that `JaxSeaquest`, and maps NEXUS’s in-training
`no_enemies` flag onto the fork’s **`no_enemies`** mod.

NEXUS targeted a much older JAXAtari. `scripts/nexus/compat.py` bridges the
gaps on the installed fork: drop `detect_prob`/`std_dev`, keep OC
`clip_reward=False`, collapse gymnasium 6-tuples → NEXUS 5-tuples on
`MultiRewardLogWrapper.step`, and unwrap modern `ObjectCentricState` for
video recording.

## Default paper training → our mod eval

```bash
# NEXUS symbolic (LLM meta + skill rewards) with paper HPs, current Seaquest:
python scripts/run_nexus.py train --alg cbl_seaquest_nexus_symbolic_paper

# Our paper_eval seaquest mods (writes outputs/models/paper_eval/eval/*_mods.json):
python scripts/run_nexus.py mod-eval --alg cbl_seaquest_nexus_symbolic_paper \
  --ckpt outputs/models/nexus/cbl_seaquest_nexus_symbolic_paper
```

Same architecture with **our** skill HPs:

```bash
python scripts/run_nexus.py train --alg cbl_seaquest_nexus_symbolic_ourhps
```

## Baselines from their paper

| Role | Alg stem | Notes |
|------|----------|--------|
| Flat PQN (shipped sea1) | `cbl_seaquest_pqn_nexus_hps` | Matches `pqn_jaxtari_sea1.yaml` |
| Flat PQN (Table 1) | `cbl_seaquest_pqn_paper_table1` | 1024 envs / 128 steps / grad 10 / ε→0.1 |
| HPQN / HRL baseline | `cbl_seaquest_hrl_baseline_paper` | `NO_REWARDS: true`, learned meta |
| HPQN + our HPs | `cbl_seaquest_hrl_baseline_ourhps` | Same flags, CBL schedule |

Upstream names (also usable via `--alg`): `pqn_jaxtari_sea3_hier_llm`,
`pqn_jaxtari_sea2_hier_baseline`, `pqn_jaxtari_sea1`, `pqn_jaxtari_sea4_hier_comb`.

## PQN hyperparameter comparison (why their PQN looks weak)

Sources: NEXUS paper Table 1; `pqn_jaxtari_sea1.yaml`; CBL
`configs/experiment/paper_eval/rq1/seaquest_pqn.yaml` (Gallici PQN Atari HPs).

| Parameter | NEXUS Table 1 | NEXUS sea1.yaml (actual) | CBL RQ1 flat PQN |
|-----------|---------------|---------------------------|------------------|
| TOTAL_TIMESTEPS | 5e7 | 5e7 | ~2.5e7 (matched skill budget) |
| NUM_ENVS | 1024 | **512** | 128 |
| NUM_STEPS | 128 | **64** | 32 |
| LR | 1e-4 | 1e-4 | **2.5e-4** |
| MAX_GRAD_NORM | 10 | **0.5** | 10 |
| LAMBDA | 0.65 | 0.65 | 0.65 |
| NUM_EPOCHS | 5 | 5 | 2 |
| NUM_MINIBATCHES | 128 | 128 | 32 |
| HIDDEN_SIZE | 64 | 64 | **128** |
| NUM_LAYERS | 3 | 3 | **2** |
| EPS start/finish/decay | 1.0 / 0.1 / 0.3 | **1.0 / 0.001 / 0.1** | 1.0 / 0.001 / 0.1 |
| EPISODIC_LIFE | (not stated) | **false** | **true** |
| Observation | object-centric MLP | object-centric | **pixel CNN 84×84** |
| FRAME_SKIP | 4 (wrapper default) | 4 | 4 |

Takeaways:

1. Their published **pixel** comparison is not apples-to-apples with Gallici/CBL
   pixel PQN. The flat image trainer is a separate CNN, and hierarchical NEXUS
   used to ignore `OBJECT_CENTRIC: false` and keep the MLP. Hierarchical runs
   now follow that flag: true keeps the object MLP; false uses the corrected
   four-frame channel-stacked CNN for the option heads and a learned selector.
   Symbolic rules still read the game state. The HRL factory accepts the same
   flag. Old object checkpoints are not pixel checkpoints.
2. sea1 **does not match Table 1** (envs/steps/grad/ε). Prefer
   `cbl_seaquest_pqn_paper_table1` when citing the paper table.
3. CBL RQ1 uses **paper-PQN (Gallici) pixel HPs** at a shorter matched budget —
   higher LR and larger hidden size; not an “overtuned NEXUS” clone.
4. Hierarchical NEXUS options use λ=0.5 (learned meta) / 0.65 (symbolic) and
   meta λ=0.9 — already noted in our HP sweeps (`lam05`, `lr1e4`).

## HRL / HPQN baseline flags

From `pqn_jaxtari_sea2_hier_baseline.yaml`:

- `HIERARCHICAL: true`, `NUM_AGENTS: 3`
- `META_POLICY: learned`, `META_GREEDY: false`
- `NO_REWARDS: true` → every option gets env reward only
- `LAMBDA: 0.5`, `META_LAMBDA: 0.9`
- Same outer PQN schedule as hierarchical NEXUS (1024×128, LR 1e-4, …)

## Smoke

```bash
python scripts/run_nexus.py train --alg cbl_seaquest_nexus_symbolic_paper --trial --wandb-mode disabled
```

## Seaquest harness repair (2026-10-23)

The three copied thesis symbolic results for seeds 44/45/46 each contain nine
errors, zero episodes, and null means. Their `axis 1 is out of bounds for array
of dimension 1` comes from `llm_meta_policy_shoot_default`: it reduces
`diver_positions[..., 2]` on axis 1 and expects `(num_envs, num_divers, 3)`.
The offline evaluator supplied a single unbatched state while supplying a batched
observation. The evaluator now batches every state leaf before invoking the same
selector used by training. The paper's main Seaquest experiment uses three
skills (combat, collect divers, surface); the four-skill appendix prompt is not
the checkpoint architecture. The existing skill order/rewards and symbolic rules
are retained.

The evaluator now runs each episode in a compiled JAX loop, records rollout
lengths and cutoff flags, and writes `status` / `failed_mods`. Any failed mod
writes diagnostic JSON/CSVs and raises an error, making the CLI exit nonzero.
`check-eval` validates requested mods, episode counts, returns, and finite means,
including for old JSONs without a status field. The generated thesis batch script
uses this validation instead of checking file existence and reuses existing
checkpoints when retrying failed evaluations.

Training also required corrections to match the supplied paper, Section 3.1,
Equations 2–3: each skill must bootstrap with its own maximum next-action Q-value,
and each lambda-return must use the value at the matching next time step. The
old trainer selected action `state_idx` at the rollout boundary and used shifted
next-state values in the recursion. `lambda_return_targets` fixes both. A minimum
in-training evaluation interval of one update allows short runs to evaluate.
These implementation changes live in the `third_party/symbolic_options`
submodule and must be included when moving the fix to another machine.

Existing checkpoints can be evaluated after the harness fix. The corrected
training targets apply to newly trained models; use a fresh save directory for
a baseline trained with the corrected implementation.

Local reproduction (the existing `cuda_jax` Python has all dependencies; the
local conda `jaxatari` environment lacks Hydra, RTPT, safetensors, and platformdirs):

```bash
PY=/home/paul/.pyenv/versions/cuda_jax/bin/python
$PY -B scripts/run_nexus.py train \
  --alg cbl_seaquest_nexus_symbolic_paper --seed 44 \
  --save-path tmp/nexus_repair_smoke/train --trial --wandb-mode disabled \
  --extra alg.TOTAL_TIMESTEPS=768 alg.TOTAL_TIMESTEPS_DECAY=768 \
  alg.NUM_EPOCHS=1 alg.NUM_MINIBATCHES=2 alg.TEST_NUM_ENVS=2 \
  alg.TEST_NUM_STEPS=64 alg.RECORD_VIDEO=false
$PY -B scripts/verify_nexus_updates.py --ckpt tmp/nexus_repair_smoke/train
$PY -B scripts/run_nexus.py mod-eval \
  --alg cbl_seaquest_nexus_symbolic_paper --ckpt tmp/nexus_repair_smoke/train \
  --seed 10044 --eval-episodes 1 --eval-num-seeds 1 --eval-max-frames 3000 \
  --mod-keys base no_enemies swap_diver_enemy_labels swap_diver_shark_roles \
  gravity faster_enemies shift_lanes random_spawns change_background_color \
  --out-json tmp/nexus_repair_smoke/final_eval/mods.json
$PY -B scripts/test_nexus_harness.py
```

The training run completed three updates (8 environments × 32 steps × 3 = 768
agent steps; six optimizer steps per skill), with clean/no-enemies training eval
enabled. Comparing the saved checkpoint with its exact random initialization
confirmed finite parameters and nonzero changes in all three skills. Evidence:
`tmp/nexus_repair_smoke/train/verification.json` and
`tmp/nexus_repair_smoke/final_eval/mods.json`. The verification also checks
hand-calculated lambda returns for lambda=0, 0.5, and 1, plus a terminal boundary.
An injected rollout failure verified diagnostic writing, exception propagation,
and rejection by `check-eval` (`tmp/nexus_repair_smoke/expected_failure/`).

These short runs establish train/eval integration, not baseline performance.
The legacy `--eval-max-frames` option still counts agent decisions (four game
frames per step); output explicitly records this unit and per-episode cutoffs.
The copied full thesis evaluations are not replaced by smoke scores. Their
referenced thesis checkpoints are not present locally; only older local NEXUS
trial checkpoints are available here.
