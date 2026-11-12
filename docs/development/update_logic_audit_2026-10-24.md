# Update-logic audit and rerun gate — 2026-10-24

Scope: the active CBL heuristic-guided skill trainer, tandem meta and skill
trainers, flat PQN, and the optional sampled-object relabeling path. Existing
worktree changes, reward semantics, action interfaces, and checkpoints are retained.
This audit establishes numerical correctness for the cases below; it does not
prove absence of all possible defects or establish learning competence.

## Reference contract

- Gallici et al., *Simplifying Deep Temporal Difference Learning*,
  https://arxiv.org/html/2407.04811v6, Eq. 18–19, Algorithm 1, Appendix B.4.
  The next value for reward r[t] is max Q(s[t+1]), not Q(s[t+2]).
  Targets are calculated before the minibatch epochs and treated as constants.
  Algorithm 1 uses the half squared-error gradient convention.
- NEXUS, *From Objects to Skills: Interpretable Meta-Policies for Neural
  Control*, local `/home/paul/Desktop/9847_From_Objects_to_Skills_In.pdf`,
  Section 3.1 Eq. 2–3. Skills have independent parameter/optimizer states,
  learn their own rewards from a shared executed transition, and bootstrap
  from their own next-action maxima. The learned selector has its own reward.
- `third_party/symbolic_options/src/symbolic_options/hierarchical_pqn_jaxtari.py`:
  the local corrected `lambda_return_targets` implements the NEXUS equations.
  The original vendored implementation has known indexing errors and is not
  an unquestioned numerical oracle.
- `docs/nexus_integration.md`, `docs/thesis_final_SPEC.md`,
  `docs/thesis_semantics_followup_runs.md`, and the thesis note
  `notes/methodology_implementation_audit_2026-10-23.md` establish the declared
  reward, terminal, target-switch, and cross-skill-update settings.

Using V[t] = max Q(s[t+1]) for a transition, the corrected recursion is

    G[t] = r[t] + gamma * (1-done[t]) * ((1-lambda)*V[t] + lambda*G[t+1])
    G[T] = V[T-1]

Environment termination, skill outcome, and the configured target-switch cuts
remain task terminals: their target is exactly the immediate reward. The
rollout boundary alone is a sampling boundary and retains bootstrapping.
The existing final `global` target-switch semantics are preserved, including
suppression of bootstrap at a switch. The last row has no observed following
selector decision; it bootstraps the retained target instead of inventing a cut.

## Corrections

1. Shared, detached lambda-return helper, used by skills, tandem and flat PQN.
   Skill paths previously used the following transition's next-state value at
   each reverse-scan iteration. Flat and meta paths had a boundary initialization
   shift in the reverse scan. These are different indexing patterns with the
   same root problem; the former claim that every flat/meta step simply used
   s[t+2] was too broad.
2. Tandem terminal transitions now retain the actually executed target. Only
   the next selector carry is reset to the no-target sentinel. This preserves
   terminal meta-action indexing, target cues, and skill-update eligibility.
3. Heuristic-guided skills honor `SKILL_LAMBDA`. Both final lambda settings
   are 0.65, so ignoring this override previously did not change those runs.
   `SKILL_REWARD_LAMBDA` is a separate reward-mixture parameter and stays 0.
4. Sampled relabel rows are matched across time by global object-slot ID.
   An absent/ineligible next row ends the sampled trace with the current
   target's one-step bootstrap; it does not use another target's future rewards.
   `involved` cuts compare actual sampled global IDs. Relabeled one-step mode
   now honors `SKILL_BOOTSTRAP=one_step`. IDs identify wrapper slots; this
   does not introduce a new persistent entity tracker beyond that interface.
5. Frozen tandem learners retain parameters and optimizer states. Multiplying
   gradients by zero previously allowed existing RAdam momentum to move weights.
6. The meta linear learning-rate schedule counts `META_NUM_EPOCHS` when it
   differs from skill epochs. Final matched-epoch configurations are unaffected.
7. User decision: use half-MSE consistently in skill, meta, and flat updates,
   matching NEXUS/PQN. This is an explicit recipe change, not part of the
   bootstrap-indexing repair. No claim is made that half-MSE outperforms MSE.
8. JAX 0.11.2 exposed tandem buffer-ownership errors: the collection JIT donated
   an RNG subsequently reused by meta training, and skill training could donate
   the meta update-counter buffer that metrics still needed. Only replaced
   train states / rollout carry are now donated. Collection, meta updates, skill
   updates and the next outer iteration receive distinct RNG streams, rather
   than reusing the collection input key after its internal RNG was discarded.
   The same fix is applied to the compiled-region execution benchmark.

## Hand calculations

Known three-step check: gamma=0.9, lambda=0.5, rewards=[1,2,3],
next_values=[20,30,40], no terminals.

    G2 = 3 + 0.9*40                       = 39
    G1 = 2 + 0.9*(0.5*30 + 0.5*39)       = 33.05
    G0 = 1 + 0.9*(0.5*20 + 0.5*33.05)    = 24.8725

If transition 1 is terminal, G1=2 and G0=1+0.9*(10+1)=10.9.
The future reward at transition 2 cannot pass through that terminal.
At lambda=0 the nonterminal targets are [19,29,39]; at lambda=1 they are
[34.39,37.1,39]. These are checked as literal expected numbers.

Three randomly generated four-step examples, NumPy seed 20261024,
gamma=0.9, lambda=0.65 (so gamma*(1-lambda)=0.315 and gamma*lambda=0.585):

| Case | Rewards | Next values | Terminal indices | Targets, t=0..3 |
|---|---|---|---|---|
| 0 | [3,2,-4,0] | [3,0,-3,5] | none | [4.3236046875,0.6471875,-2.3125,4.5] |
| 1 | [2,1,1,-4] | [-5,7,-2,1] | 1 | [1.01,1,-1.4435,-3.1] |
| 2 | [2,-2,3,0] | [9,10,-4,2] | 3 | [6.1032215,2.1679,1.74,0] |

Case 0 explicitly:

    G3 = 0 + 0.9*5                   = 4.5
    G2 = -4 + 0.315*(-3) + 0.585*4.5 = -2.3125
    G1 = 2 + 0.315*0 + 0.585*(-2.3125) = 0.6471875
    G0 = 3 + 0.315*3 + 0.585*0.6471875 = 4.3236046875

Case 1 has G1=1, hence G0=2-1.575+0.585=1.01. Later rewards are excluded.
Case 2 has G3=0, hence G2=3-1.26=1.74,
G1=-2+3.15+0.585*1.74=2.1679, and
G0=2+2.835+0.585*2.1679=6.1032215.

The automated independent oracle enumerates forward n-step returns with exact
`Fraction` arithmetic and their geometric mixture weights; it does not repeat
the production reverse recursion. Random tests cover time lengths 1,2,5,32,
multiple environments, objects and skills, and lambda=0,0.5,0.65,1.

A seeded sample from the actual corrected-fork skill rollout has rewards
[0,0,0,0], no terminals, and next values all equal to 1.2142575979232788.
Using the runtime float32 gamma/lambda (approximately 0.99/0.65), scalar
arithmetic gives G3=1.2021150335241089, G2=1.194301293544579,
G1=1.1892731520037463, G0=1.186037543009733. The compiled targets are
[1.1860374212265015,1.1892731189727783,1.1943012475967407,1.2021150588989258],
agreeing at float32 precision. Sparse terminal/reward cases are supplied
explicitly by the synthetic hand calculations above; a short real rollout
alone would not adequately test those boundaries.

The accompanying `update_logic_audit_2026-10-24_runtime_samples.json` retains
seeded samples from all nine real training arms, including per-object IDs for
relabeling, the environment versions, and SHA-256 hashes of the audited source.

Relabel test: rows [A,B], [B,A], [A,B], with B's rewards and values ten times
A's, must produce rows [24.8725,248.725], [330.5,33.05], [39,390]. If A is absent
at the next sampled step, its first target is 1+0.9*20=19 even when another row's
next reward is 999. Invalid future rows are likewise excluded from the trace.

For half-MSE, each eligible tabular output has derivative (Q-G)/N and all
unexecuted action outputs have derivative zero. Tests execute the actual nested
trainer loss functions with known tables and compare all gradient entries,
including cross-skill and selected-skill-only masks and all-invalid meta rows.

RAdam check with clipping norm 2 and learning rate 0.1: gradient [3,4] becomes
[1.2,1.6]. From parameters [1,-2], the first unrectified RAdam step gives
[0.88,-2.16]. A second raw gradient [-3,0] clips to [-2,0], giving first moment
[-0.092,0.144] and bias correction 1-0.9^2=0.19. The second parameters are
[0.9284210526,-2.2357894737]. Both steps are checked numerically. A freeze test
starts with nonzero optimizer momentum and checks that every state leaf is
unchanged when frozen.

## Verification and boundaries

- `tests/test_lambda_returns.py`: literal arithmetic, independent forward-view
  random oracle, terminal/reset values, target detachment, shape checks,
  resampled/permuted object identities and unavailable future rows.
- `tests/test_bootstrap_wiring.py`: executes the actual target construction
  blocks with known Q-functions and compares results to the independent oracle;
  tests environment, outcome and target-switch cuts, own-skill Q values,
  masked meta candidates, flat next-state indexing, one-step mode, and terminal
  executed-target retention.
- `tests/test_update_arithmetic.py`: actual production losses, analytic action
  gradients (including shared linear weights across repeated actions), masks,
  minibatch alignment, clipping/RAdam, frozen learners, and distinct/reproducible
  per-region RNG streams for both typed and legacy JAX keys.
- `tests/test_trajectory_cuts.py` and `tests/test_schedules.py`: switch masks and
  the meta epoch override, in addition to existing regression cases.
- `tests/test_update_training_audit.py`: two short actual updates per arm from
  final Seaquest configurations, with runtime target arrays checked against the
  independent forward-view oracle. Covers ego, pixel, OC skills; OC/pixel tandem;
  sampled-K relabeling in both trainers; OC/pixel flat PQN. These are CPU/JIT
  integration checks, not learning-competence experiments.
- Existing smoke tests also exercise Phoenix and Space Invaders skills, ego
  embedding, and tandem compilation.

Broader-suite failures outside the modified update code remain: a BeamRider
slot-count assertion still expects five DESTROY slots although the documented
current layout includes three laser-scored coins (eight slots); the episodic
metric test expects a per-life score under a key deliberately defined as
full-game return; three Montezuma observation/heuristic fixtures disagree with
current behavior; and the `oc_embed` test fails when truncating six slots to
four. Montezuma and `oc_embed` are excluded from the final recipe. The last two
issues concerning episodic metrics and OC embedding were also reproduced on
the unchanged implementation before this patch. These failures are recorded
rather than treating the entire project test suite as green.

The bootstrap helper regression's fake network had an outdated signature
without `oc_psi`. Its stub is corrected to accept the actual network protocol,
and its chunked/full-forward comparison now passes.

## Environment provenance

The initial `cuda_jax` interpreter (JAX 0.7.0) imported the main
`/home/paul/Documents/JAXAtari/src/jaxtari` checkout. Its missing `no_cars` and
`no_enemy_fire` registrations caused two failures. These were import-provenance
errors, not missing implementations in the robustness fork. Both tests pass
when the robustness fork is placed first on PYTHONPATH, even with cuda_jax.

Primary final verification uses:

- Python: `/home/paul/.pyenv/versions/jaxtari-fork/bin/python`.
- Imported module: `/home/paul/Documents/JAXAtari-seaquest-robustness/src/jaxtari/__init__.py`.
- JAX/jaxlib 0.11.2, Flax 0.12.9, Optax 0.2.8, CPU backend.
- Robustness fork base commit: `dcd2ebfae44bae61306538ecf70f5b5a84b90ab9`.
- CBL base commit: `ed5dc9ff315297bba245c084b836c77eebc97191`, plus existing
  user changes and this patch (the dirty worktree is intentional).

The final numerical checks also compare against the corrected NEXUS helper
directly, without importing its older game dependencies. The synthetic freeze
test uses a parameter dictionary like real Flax networks so it works with both
the older and newer Flax TrainState implementations.

Reproduce the focused correctness checks from the CBL repository root:

```bash
env PYENV_VERSION=jaxtari-fork \
  PYTHONPATH=/home/paul/Documents/JAXAtari-seaquest-robustness/src:src \
  JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -B -m pytest -q -p no:cacheprovider \
  tests/test_lambda_returns.py tests/test_update_arithmetic.py \
  tests/test_bootstrap_wiring.py tests/test_trajectory_cuts.py \
  tests/test_schedules.py tests/test_skill_bootstrap.py
```

Use the same environment prefix with `tests/test_update_training_audit.py` for
the nine compiled training checks. Set `CBL_UPDATE_AUDIT_REPORT_DIR` to save
seeded runtime traces and targets as JSON. Assert interpreter and module paths
on the training machine before starting the competence pilots.

Final focused gate in jaxtari-fork: **91 passed**, no skips. This includes
comparison with the corrected NEXUS return helper and both JAX key formats.
The nine compiled training arms all passed, checking **66 runtime return
batches**. The four skill arms completed before the tandem-only buffer/RNG
correction; the three tandem and two flat arms completed after it. The skill
update source did not change between those runs.

The broader non-slow CBL run recorded **692 passed, 6 failed, 18 skipped,
21 deselected** after correcting the Flax test fixture and environment import.
The six failures are the unchanged non-update cases listed above. Two additional
RNG-isolation tests were subsequently added and are included in the 91-test
focused gate. Invoking unrestricted pytest from the root also collects legacy
vendored JAXAtari tests; those require old `wrappers_new` dependencies and are
not part of the CBL correctness command.

The archived `_legacy/train_meta_agent.py` still contains the old algorithm.
It is not the active training entry point. Do not use it for corrected runs.
The NEXUS baseline must use its already-corrected vendored trainer, not an
uncorrected upstream checkout. Saved checkpoints encode past training and are
not repaired by changing these routines.

## Rerun gate

Use the declared learning rate 2.5e-4, gamma=0.99 and lambda=0.65 with half-MSE.
No new learning-rate sweep is required solely by the loss convention change.
After correctness validation, run one training seed for each core learner
(skills, tandem, flat PQN) to check competence before committing all seeds.
Only instability or clear acquisition failure warrants a small learning-rate
pilot. Keep robustness/transfer test outcomes out of tuning decisions.

Give corrected runs a distinct output/version identifier; preserve old artifacts
for provenance. Retrain learned checkpoints used by the final comparisons and
repeat their modification/transfer evaluations. Random-action and fixed-rule
controls do not depend on training targets. Do not mix corrected NEXUS training
with old-target CBL runs as if they followed the same update recipe.
