# Breakout launch fix

The common environment adapter sends FIRE for one native simulation frame when
Breakout has lives remaining and `game_started` is false. Requested actions resume
on the following native frame, including within a repeated-action step. The
adapter does not reset the game or intervene during a live rally. It applies to
hierarchical agents and flat PQN training and evaluation.

The native OC ball record was active while the renderer hid the unlaunched ball.
The adapter now marks that ball inactive, and the semantic wrapper preserves its
activity. The launch flag is used only by the adapter and is not added to policy
observations. Automatic launch is part of the environment control protocol.
Training action records remain the policy's requested actions under this protocol.

## Scheduler handoff

Use A's source training recipe with early collection and `centre_ray` geometry.
This fix does not change Seaquest training, rewards, geometry or Pong.
Existing Seaquest checkpoints can be reused for Breakout transfer evaluation.
Rerun previously completed Breakout evaluations with the fix. Use this protocol
consistently for Breakout baselines and comparisons as well. Existing native
Breakout checkpoints can be evaluated, but were trained under the earlier launch
protocol, which should be recorded if they are compared with newly trained ones.

The fix is enabled by default only for Breakout. For explicit configuration use

```yaml
BREAKOUT_AUTO_LAUNCH: true
```

Set it to false to reproduce the old launch protocol. Eval override merging
preserves this setting. No cluster files or jobs were changed by this task.

Files to synchronize to the scheduler checkout

- `src/concept_based_learning/environments/breakout/auto_launch.py`
- `src/concept_based_learning/environments/breakout/breakout_label.py`
- `src/concept_based_learning/environments/env_manager.py`
- `src/concept_based_learning/baselines/flat_pqn.py`
- `src/concept_based_learning/baselines/flat_pqn_train.py`
- `src/concept_based_learning/paper_eval/config_merge.py`
- `scripts/run_paper_eval.py`

## Validation

Seven tests passed on the local RTX 5070 across
`tests/test_breakout_auto_launch.py` and `tests/test_flat_pqn_reward_clip.py`.
They cover initial launch, relaunch after actual life loss, no forced launch at
game over, unchanged actions during live play, a single FIRE pulse within four
repeated frames, visibility, configuration scope and existing flat reward clipping.

The frozen A seed 44 checkpoint was evaluated locally for episode seeds 10044,
20044 and 30044. Returns were 19, 12 and 16. All episodes reached game over after
five lost lives in 603, 536 and 679 decisions. There were respectively 2, 2 and 0
observed waiting decisions, with no consecutive waiting decisions. The standard
greedy evaluation runner independently reproduced the first return of 19.
These are launch smoke checks, not a replacement for the full evaluation suite.

Trace artifacts and the reproduction script are in
`tmp/breakout_autolaunch_A_20261027/`.
