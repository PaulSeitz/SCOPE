# Scheduler orders: finalize the raw-image slot meta-policy

The author selected `META_ARCHITECTURE=pixel_slots` after the Seaquest seed-44
feasibility pilot. **Prepare execution when resources permit; this handoff launches
nothing, edits no active queue, and cancels no running process.** Do not run the
candidate-marked fallback or start further legacy hybrid-pixel jobs by default.
The remaining OC representation decision is separate: do not change OC jobs or
configs on the basis of this handoff.

## Accepted pilot and first action

User-reported / locally logged result on cluster2: mean Seaquest base return
25211 +/- 7118 over nine episodes, range 14070--37400, mean rescues 2.4, rack
reaches six, approximately 557 kills. This is one trained seed; confirm whether
7118 is episode standard deviation from the raw JSON. Do not label it across-seed
uncertainty. The result establishes source-task feasibility, not robustness.

1. Pull the pilot JSON, summary, episode data, saved configuration, and BOTH
   skill/meta checkpoints from cluster2. Preserve the `pm_seaquest_slots_seed44`
   provenance and save the actual source revision/configuration snapshot.
2. Verify the saved architecture is `pixel_slots`, with four unmarked grayscale
   84x84 frames and NO candidate-feature, label, event or target-lock inputs.
   Match the final skill/reward/selector/action-frame/training settings.
3. If compatible, reuse that checkpoint for its full home modification evaluation
   and frozen-skill transfers using the provided `ps_...` eval-only aliases. The
   chain runner checks required artifacts and critical saved-config fields before
   `--reuse-pilot --run`. Also compare any later interface/geometry flags and
   preprocessing choices manually; this check is not a substitute for provenance.
   Do not rename or overwrite the pilot checkpoint. If settings differ, train
   final Seaquest seed44 afresh under its distinct `tf_...pixel_slots...` stem.

## Execution priorities

1. Seaquest seeds **45 and 46**: new raw-slot tandem trains, then complete-agent
   home modification evaluation and skill-only transfers.
2. Seaquest seed **44** full home evaluation and transfers from the accepted pilot
   if compatible (otherwise the fresh final seed44 chain).
3. Name This Game seed **44** with the same raw-slot architecture, then seeds
   **45 and 46**, each followed by home evaluation and skill-only transfers.
   Report acquisition failure if it occurs rather than silently selecting a
   different architecture for that environment.
4. Keep the marked-candidate fallback unrun unless the author explicitly reopens
   the architecture decision. Do not spend capacity on it just because it exists.

Priorities are recommendations, not a GPU assignment. Leave existing jobs alone;
use the scheduler's next available slot. Both selector designs were locally
smoke-tested; use the raw-slot benchmark on the actual GPU for resource estimates
if needed. No new runtime estimate is inferred from the competence result.

## Artifact families and coverage

`configs/experiment/paper_eval/pixel_meta_final/` contains 60 final configs:
6 trains (two environments x seeds44/45/46), 6 complete-agent home evals, and
48 frozen-skill transfers (each source/seed to eight other suite games). Another
9 configs provide eval-only reuse of the Seaquest44 pilot: one full home eval and
8 transfers. Select the reused or fresh seed44 path; never count both as different
training seeds or automatically run the duplicate train.

Training names: `tf_{env}_tandem_pixel_slots_seed{seed}`.
Home eval: append `_eval`.
Transfers: `tf_zs_tandem_pixel_slots_skills_{source}_to_{target}_seed{seed}`.
Time Pilot is excluded, matching the current suite. Modifications are exactly the
current recipe's declared set; Breakout currently has base-only evaluation.

The final clone preserves the current tandem training budget, three ego skills,
repaired rewards, frame conventions, exploration and training settings. It changes
only the pixel selector architecture and artifact family. Existing
`tf_*tandem_pixel*` hybrid configs and checkpoint paths remain unchanged and must
not be mixed into this comparison. The raw-slot model encodes the scene once;
output heads identify target slots, not skill classes. Structured validity masks,
virtual targets and semantic skill routing remain outside its network.

Home robustness evaluates the complete tandem agent (both source selector and
skills frozen). Transfer extracts only the learned skills and supplies the target
environment's heuristic selector, labels and action mapper. It is not transfer of
the pixel meta-policy. Verify this distinction when labeling plots.

## Commands

Sync the new generator, chain runner, pixel selector implementation/factory, and
all `pixel_meta_final` YAMLs to the execution host. Do not rerun the full
`materialize_thesis_final.py` while existing queue jobs may still open its files.
The isolated generator refuses to overwrite changed final configs.

The chain runner prints the exact commands by default. Add `--run` only when the
scheduler allocates a slot; set CUDA_VISIBLE_DEVICES through the ordinary cluster
policy. It runs train -> home eval -> eight skill-transfer evaluations and stops
on failure. `--resume` reuses completed artifacts; verify their identity first.

```bash
# Preview remaining-seed chain; append --run to execute on an allocated GPU.
python scripts/run_pixel_meta_final_chain.py --env seaquest --seed 45
python scripts/run_pixel_meta_final_chain.py --env seaquest --seed 46

# After pulling and checking pilot artifacts; no training step in this chain.
python scripts/run_pixel_meta_final_chain.py --env seaquest --seed 44 --reuse-pilot

python scripts/run_pixel_meta_final_chain.py --env namethisgame --seed 44
python scripts/run_pixel_meta_final_chain.py --env namethisgame --seed 45
python scripts/run_pixel_meta_final_chain.py --env namethisgame --seed 46
```

The complete seed44 pilot reuse home-eval stem is
`ps_seaquest_slots_seed44_full_eval`. The `ps_zs_slots_seaquest_to_*_seed44` aliases
retain that pilot source checkpoint and architecture reference without copying it
into the final namespace. Selection of the architecture used pilot seed44: disclose
its reuse if it appears in final summaries. Seeds45/46 are independent
replications of the selected architecture, not evidence that architecture selection
was performed without pilot data.

When finished, pull per-episode results, update Have/Pipeline, report individual
training-seed means and a separate across-seed summary, and retain target-choice,
collection/rescue and life-loss diagnostics. Do not promote the provisional pilot
number into a finalized multi-seed thesis result.
