# ALE PQN (`f163gauq`) ↔ JAXAtari flat PQN match notes

Source W&B: https://wandb.ai/jaxatari/baselines_new/runs/f163gauq  
Code: `scripts/benchmarks/baselines/pqn_atari.py` @ `k4ntz/JAXAtari@865bac70`  
Local copies: `source_wandb_f163gauq_*.yaml/json`, `source_pqn_atari.py`

## Hyperparameters (exact map)

| CleanRL / W&B | Our flat_pqn YAML | Value |
|---|---|---|
| `total_timesteps` | `TOTAL_TIMESTEPS` | **10_000_000** → 2441 updates |
| `num_envs` | `NUM_ENVS` | 128 |
| `num_steps` | `NUM_STEPS` | 32 |
| `update_epochs` | `NUM_EPOCHS` | 2 |
| `num_minibatches` | `NUM_MINIBATCHES` | 32 |
| `learning_rate` / `anneal_lr` | `LR` / `LR_LINEAR_DECAY` | 2.5e-4 / false |
| `max_grad_norm` | `MAX_GRAD_NORM` | 10 |
| `gamma` | `GAMMA` | 0.99 |
| `q_lambda` | `LAMBDA` | 0.65 |
| `start_e`→`end_e` / `exploration_fraction` | `EPS_*` / `EPS_DECAY` | 1→0.001 / 0.1 |
| `seed` | `SEED` | **2** |
| optimizer | (hardcoded) | RAdam |
| loss | (hardcoded) | MSE on Q(λ) returns |
| `episodic_life` | `EPISODIC_LIFE` | true |
| sticky (`repeat_action_probability=0`) | `STICKY_ACTIONS` | false |
| `use_fire_reset` | `FIRST_FIRE` | true |
| `noop_max=30` | `NOOP_MAX` (default) | 30 |
| `frameskip`/`stack_num` | `FRAME_SKIP`/`FRAME_STACK_SIZE` | 4 / 4 |
| reward clip wrapper | `CLIP_REWARD` | true |
| 84×84 gray stack | `PIXEL_*` + `USE_NATIVE_DOWNSCALING=false` | ALE-style resize |

ε decay: CleanRL anneals over `0.1 * total_timesteps` env steps; we anneal over
`0.1 * NUM_UPDATES` updates (= same fraction of agent steps).

## Architecture

**Matched (Nature CNN + LN + 512 → |A|):**

```
obs/255 → Conv 32@8×8/s4 → LN → ReLU
        → Conv 64@4×4/s2 → LN → ReLU
        → Conv 64@3×3/s1 → LN → ReLU
        → Flatten → Dense 512 → LN → ReLU → Dense |A|
```

`HIDDEN_SIZE` / `NUM_LAYERS` are OC-only; pixel path ignores them (same as CleanRL).

**Intentional / framework diffs (kept for jaxatari ckpt compatibility):**

| | CleanRL `pqn_atari.py` | Our `flat_pqn.QNetwork` (pixel) |
|---|---|---|
| Layout | NCHW | NHWC |
| Conv/Dense init | orthogonal(√2) | `he_normal` |
| LayerNorm after conv | feature-map LN `[C,H,W]` | channel-only LN (last axis) |
| Extra BN | none | dummy `BatchNorm` (unused; ckpt key compat) |
| Param count (Seaquest 18 acts) | ~1.737M | ~1.695M (LN param gap) |

Channel-only LN matches jaxatari / purejaxql Flax PQN; changing it would break
existing paper_eval flat PQN checkpoints.

## Env domain

| | ALE `AtariVectorEnv` | JAXAtari wrap stack |
|---|---|---|
| Simulator | Stella / ale_py | jaxatari Seaquest |
| Resize | built-in 84×84 | `PixelObsWrapper` + `USE_NATIVE_DOWNSCALING=false` |
| Action space | minimal (18 for Seaquest) | minimal 18 |

This suite trains the **same HPs / same CNN topology** on **JAXAtari Seaquest**
(domain gap vs the ALE W&B run is intentional).
