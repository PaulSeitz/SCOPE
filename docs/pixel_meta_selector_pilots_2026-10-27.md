# Scheduler handoff: pixel-only meta-selector pilot

**Decision update:** the author accepted the raw-image slot selector after the Seaquest seed-44 pilot (25211 +/- 7118, episode aggregation pending artifact verification). Candidate-marked fallback is on hold. Current execution orders: [pixel_meta_finalization_2026-10-27.md](pixel_meta_finalization_2026-10-27.md). The setup and conditional plan below are retained as historical provenance.

Prepared at the thesis author's request. **Recommendation only: no jobs launched,
no queue scripts changed, no running job interrupted.** The scheduler/author chooses
when to execute. Local run records do not establish whether a remote pixel-tandem
job has already imported its network code; verify remote processes first.

## Decision and execution order

1. Run the architecture tests and a tiny training smoke. Benchmark both selectors
   on the proposed GPU before reserving a full pilot slot, especially candidates.
2. Train **Seaquest raw-slot seed 44** (`pm_seaquest_slots_seed44`), then run its
   base-only evaluation (`pm_seaquest_slots_seed44_eval`). Keep ordinary training
   metrics enabled: do not use `--lean` for this diagnostic run.
3. Review source-task acquisition and target-slot correspondence. If the simple
   selector demonstrates useful behavior with consistent target choices and
   acceptable runtime, retain it and run the Name This Game raw-slot pilot.
4. If raw-slot learning is poor, run the **matched Seaquest candidate-marked pilot**
   after its throughput/memory benchmark. Use the same seed, training budget,
   skill architecture/rewards, exploration and base-evaluation protocol. If needed,
   repeat this matched pair in Name This Game.
5. Choose the final architecture using source-task feasibility, not final
   robustness/transfer scores. Then run the final independent training seeds and
   held-out modification/transfer protocol. One seed establishes feasibility,
   not superiority. Retain both pilot outcomes and the selection rationale.

Before starting, record the chosen minimum acceptable base competence and resource
ceiling. Use a task-grounded criterion: repeatedly collecting divers and surfacing
as well as shooting in Seaquest, rather than score from shooting alone. Compare
with the existing ego/OC tandem and random/fixed controls where available. Do not
choose a numeric pass threshold after inspecting these pilot results. Failure of
a raw-slot head is not proof that candidate marking is necessary.

## Architecture contracts

- `pixel_slots`: original grayscale image stack -> shared CNN/MLP -> one Q-value
  per target slot including NOOP. No raw object data, labels, event flags, task
  scalars or target-lock scalar enters the network. Slot ordering comes from the
  wrapper; unlike pixels, internal slot identity may be unobservable when visually
  identical objects exchange slots. Inspect target selections against slot/box
  overlays before interpreting low return as a learning-algorithm failure.
- `pixel_candidates`: shared scalar CNN/MLP applied to a marked image stack for
  every padded candidate slot. Only bounding boxes and activity are used to paint
  the candidate. The latest box marks every frame, matching pixel-skill marking.
  No labels, event flags, object state or visual IDs are appended to the network.
  NOOP uses the unmarked scene with a black 6x6 top-left marker and a white 4x4
  inset on every frame. Virtual waypoints use the wrapper-supplied box, so this
  design explicitly receives a visual candidate annotation beyond raw pixels.
- Candidate scoring shares parameters and uses a rematerialized scan, default
  chunk size 1. Work scales with **MAX_OBJECTS**, not just active visible objects;
  inactive slots are masked outside the network. Chunking limits activations but
  does not eliminate the cost of scoring all candidate images.
- Existing validity masks and label-to-skill routing remain outside both networks.
  Thus `pixel_slots` is a pixel-only **network input**, not removal of object-based
  candidate enumeration, masking, virtual targets, skill routing or skill inputs.
- Both variants explicitly stack frames as channels with a transpose preserving
  spatial order. Existing hybrid/legacy pixel paths are unchanged.
- The previous `per_object_mlp` + `pixel_embed` branch remains a separate **hybrid**
  pixel/object-feature architecture and still consumes its historical inputs.
  Do not present its checkpoints as these pixel-only pilots. Scheduler should
  hold unstarted hybrid-pixel final comparisons pending the architecture choice;
  do not cancel an already-running process without the author's decision.

## Configs and matching

Eight configs are under `configs/experiment/paper_eval/meta_selector_pilots/`:
`pm_{seaquest,namethisgame}_{slots,candidates}_seed44.yaml` and matching `_eval.yaml` files.
They clone the current same-environment tandem pixel seed-44 templates. Training
budget is the existing 50M nominal steps, 1024 envs x 32 rollout decisions, with
the same three ego skills, repaired interaction rewards and exploration schedule.
Only selector architecture and pilot artifact names change; both pilot CNNs use
the same channels, kernels, projection and hidden trunk. The slot head has one
output per slot; the candidate scorer shares one scalar head, so parameter counts
differ slightly and are reported by the benchmark. Regenerating this pilot after
reference-config changes is an explicit operation; the materializer refuses to
overwrite differing pilot files.

All checkpoint paths and W&B names are new. Eval YAMLs pin architecture to their
matching pilot train config and select `base` only. Existing final-suite configs,
results and automatic queues are unchanged. These pilots are not in SUITE_STEMS;
use the exact `--only` commands below. Avoid unfiltered `run_paper_eval.py` calls,
which discover all runnable configs, including pilots.

## Commands (from ConceptBasedLearning root)

Sync these files to the execution host first: `networks/pixel_meta_network.py`,
`networks/meta_factory.py`, all pilot YAMLs, the three pilot scripts and tests.
Use the cluster's regular project Python environment and GPU assignment policy.
The benchmark is synthetic forward/backward work, **not a training or performance
result**, and does not measure environment or full-trainer overhead.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python -B -m pytest -q tests/test_pixel_meta_pilots.py tests/test_meta_oc_simple.py tests/test_meta_target_mask.py

PYTHONPATH=src python -B scripts/smoke_pixel_meta_pilots.py configs/experiment/paper_eval/meta_selector_pilots/pm_seaquest_slots_seed44.yaml --output-root /tmp/pixel_meta_smoke_slots
PYTHONPATH=src python -B scripts/smoke_pixel_meta_pilots.py configs/experiment/paper_eval/meta_selector_pilots/pm_seaquest_candidates_seed44.yaml --output-root /tmp/pixel_meta_smoke_candidates

PYTHONPATH=src python -B scripts/benchmark_pixel_meta_pilots.py configs/experiment/paper_eval/meta_selector_pilots/pm_seaquest_slots_seed44.yaml --batch-size 8
PYTHONPATH=src python -B scripts/benchmark_pixel_meta_pilots.py configs/experiment/paper_eval/meta_selector_pilots/pm_seaquest_candidates_seed44.yaml --batch-size 8

python scripts/run_paper_eval.py --only pm_seaquest_slots_seed44 --phase train --abort-on-failure
python scripts/run_paper_eval.py --only pm_seaquest_slots_seed44_eval --phase eval --abort-on-failure
```

Conditional fallback:

```bash
python scripts/run_paper_eval.py --only pm_seaquest_candidates_seed44 --phase train --abort-on-failure
python scripts/run_paper_eval.py --only pm_seaquest_candidates_seed44_eval --phase eval --abort-on-failure
```

For Name This Game replace `seaquest` with `namethisgame`. Increase benchmark batch
sizes towards the real rollout batch (1024) and gradient minibatch (128); record
GPU model, memory, compile time, throughput and chunk size. If the candidate
variant requires fewer parallel environments, rerun the raw-slot comparator with
that same setting rather than treating differently configured runs as matched.
No GPU runtime estimate is asserted before this measurement. Changes of batch or
chunk configuration should use new config/artifact stems, not overwrite a pilot.

Record the actual commands, source revision, config snapshots, artifacts, slot
identity observations and accept/reject decision here or in the results log.
No final thesis comparison should use these outcomes as independent final seeds
without disclosing their role in architecture selection.

## Local validation before scheduler handoff

- 26 tests passed on CPU: feature isolation, proper frame/channel ordering,
  candidate-permutation behavior, distinct NOOP marking, chunked-score/gradient
  equivalence, nonzero finite gradients, matched pilot configs, and existing OC
  and target-mask regression checks.
- Both architectures completed two tiny real-environment Seaquest tandem updates
  and saved separate skill/meta checkpoints with reduced networks and two envs.
  These are wiring checks, not acquisition results.
- The candidate scorer also completed a JIT forward/backward check at the full
  configured 84x84 input, 35 slots and the normal network widths, batch size 1,
  on CPU. GPU memory and runtime remain unmeasured.
- No full pilot, final evaluation, cluster sync, or scheduler submission was run.
