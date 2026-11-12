# Absolute-target campaign launched 2026-11-11

Started 17:18–17:19 CET, detached in tmux session `PS_abs_target_20261111` on each host.
All ten 50M-budget training runs and all 70 evaluations completed successfully.
The last evaluation finished on cluster2 at 17:53:45 CET; cluster1 finished at
17:35:28 CET. Validated results and checkpoints are now imported into the active
`tmp/thesis_final_results/frozen_2026-11-07/` figure basis.
The usual 1,525 complete updates yield 49,971,200 environment transitions,
matching SCOPE's treatment of the remainder of its 50M budget.

Both hosts passed a CUDA train → safetensors reload → shifted-lanes evaluation
preflight using the exact deployed code. The original local 100,352-step smoke
also passed before this deployment. Preflight results are not production results.

## Placement

| Host | GPU | Runs |
|---|---|---|
| cluster1 | 0 | NTG 44 |
| cluster2 | 1 | NTG 45 |
| cluster2 | 2 | NTG 46 |
| cluster2 | 3 | NTG 47 |
| cluster2 | 4 | NTG 48 + Seaquest 48 |
| cluster2 | 5 | Seaquest 46 + Seaquest 47 |
| cluster2 | 6 | Seaquest 44 |
| cluster2 | 7 | Seaquest 45 |

Cluster2 GPU 0 acquired another user's training job after the initial availability
check, so NTG 44 moved to cluster1 GPU 0. Other occupied GPUs remain untouched.
Ollama is the allowed shared process on cluster2. Training pool fraction is 0.35
per process with preallocation disabled. CPU affinity bounds compilation to five
logical CPUs per training process. W&B logs are offline and saved locally; metrics
also stream to each model directory's `training_metrics.csv`.

## Automatic evaluation

A host coordinator launches evaluation jobs as its checkpoints become ready and
GPUs finish training. It never starts evaluation on a GPU still training a campaign
run. Before launching any worker, it checks for foreign compute processes; Ollama
is allowed. Cluster2 Seaquest evaluations only use GPUs 4–7. Up to three eval jobs
share a high-memory card, each at memory fraction 0.078 and three CPU cores;
cluster2 GPUs 1–3 run one NTG eval at a time. No transfer jobs are scheduled.

Each checkpoint gets all seven home conditions (70 jobs total). Every condition
runs three episodes on each of three evaluation seeds, using `10000 + train_seed`
as the base seed and 27,000 agent steps as the episode cap, matching the reference
archive. The final aggregated JSON is written only after all seven conditions
pass the nine-finite-returns check. Workers fail visibly in coordinator status;
failures are not silently replaced with scores or automatically retrained.

## Locations and monitoring

The isolated campaign root on BOTH hosts is:

`/home/pseitz/ConceptBasedLearning/outputs/abs_target_campaign_20261111`

Within it:

- `logs/coordinator.log`: starts, finishes and GPU occupancy waits.
- `logs/*__train.log`, `logs/*__{condition}.log`: per-worker logs.
- `status/{cluster1,cluster2}.json`: running jobs, PIDs, completed jobs and failures.
- `outputs/models/paper_eval/abs_target_ablation/`: trained models and config sidecars.
- `eval_jobs/`: individual condition JSONs and CSVs.
- `eval/`: canonical `tf_{game}_skills_ego_abstarget_seed{seed}_eval_mods.json` files.
- `preflight/smoke_passed.json`: successful GPU preflight record.
- `wandb/`: offline training run logs.

From the local CBL repository:

```bash
python scripts/status_abs_target_campaign.py
ssh cluster2 'tail -n 30 ~/ConceptBasedLearning/outputs/abs_target_campaign_20261111/logs/coordinator.log'
ssh cluster1 'tail -n 30 ~/ConceptBasedLearning/outputs/abs_target_campaign_20261111/logs/coordinator.log'
```

Implementation: `scripts/run_abs_target_campaign.py`. Do not start a second
coordinator in the same campaign directory; it is a launch-time scheduler, not a
restart/resume tool. Existing cluster working trees were not modified. The source
snapshot is captured in `source.tar.gz` on each host and locally under
`tmp/abs_target_campaign_20261111/` (with placement and manifest JSONs).

Source archive SHA256:
`5cb7bd745514f8361eb0aa59df6dd47988c245ebaee4915f466460c1591b96ae`.

After completion, collect both hosts' `eval/` directories into a fresh local result
directory and import with the thesis snapshot builder's `--abs-target-results`.
Do not overwrite an existing frozen snapshot or mix preflight data into results.

## Launch correction

The initial production attempt exposed a launcher setting absent from the smoke:
`JAX_PLATFORMS=cuda` disabled the CPU backend needed by metrics/RTPT host callbacks.
All ten jobs failed at the initial update; no completed training checkpoints were
written. Initial logs/status are preserved under `attempt1/` on both hosts.
The corrected launcher uses `JAX_PLATFORMS=cuda,cpu`, keeping CUDA as the default.
The full campaign was restarted. No input features or training hyperparameters
changed. The archive above is the initial snapshot; the corrected snapshot is
`source_launch_v2.tar.gz` with the hashes in local `manifest.json`.

Corrected launcher SHA256:
`e44553223e55c64563a8e161156733f9111c613b8afa1c1f524f7db7782982e4`.

Corrected source archive SHA256:
`e9237df817cec602e068487c64b445091cbdd2a80f52eb947ab3520071b8f441`.

Startup health verified at 17:23 CET after the launch correction: all ten jobs
finished compilation and wrote training metrics, with roughly 1.0–8.5M transitions
per run observed. The corrected coordinators report no failures. Cluster2
training cards showed 89–100% utilization and at least 22 GiB free memory.

## Completed and imported

Validated all 630 episode returns and means, all ten finite 36-input checkpoints,
checkpoint hashes, skill-input sidecars, and final training counters (1,525 updates,
49,971,200 transitions per run). Both corrected coordinators finished without failures.
The initial failed logging attempt remains archived separately on the clusters.

Imported ten new canonical evaluation JSONs, matching 70-row-total summary CSVs
and 630-row-total episode CSVs, plus all checkpoints/configs/training metrics,
into `tmp/thesis_final_results/frozen_2026-11-07/`. No existing evaluation scores
were overwritten. Provenance: `ABS_TARGET_2026-11-11.json` inside that snapshot.
The thesis `data.Runs` reader confirms five seeds for every new arm/condition.

Full results: `tmp/thesis_final_results/abs_target_2026-11-11/RESULTS.md`;
numerical comparisons (including the existing G arm): `comparison.csv` there.
Raw cluster aggregates, per-condition jobs, status and logs remain under `pulled/`.
Top-level mod lists and base-drop fields were rebuilt during import; episode
returns and means are unchanged from the original per-condition artifacts.

Seaquest absolute-target base mean is 21,317 versus SCOPE's 46,265. Absolute-target
retention is 72.0% on shifted lanes and 91.3% on random spawns; SCOPE is 61.6% and
68.2%, respectively. NTG base is 14,019 versus 16,413; shifted-tentacle retention
is 82.9% versus 90.1%. This does not show the anticipated Seaquest layout collapse;
behavior should be inspected before making a causal claim from these comparisons.

## Additional two-way transfer check (2026-11-11)

At the user's later request, all five source checkpoints were evaluated on the
other primary game's base environment: ten jobs, 90 episodes, no retraining.
All completed successfully on cluster2, started 21:58 CET; each job took about
48–75 seconds. Exact SCOPE transfer configs supplied target-game heuristic,
mapper and evaluation settings, with skill input changed to `ego_abs_target`.

Seaquest → NTG: 702 ± 354 (SCOPE reference 1,708 ± 2,614).
NTG → Seaquest: 29 ± 26 (SCOPE reference 130 ± 89).
Mean ± sample SD is across five checkpoint means, each from nine episodes.
These are 5.0% and 0.14% of the respective native absolute-target means.

Results and CSVs were validated and appended to the same frozen snapshot under
`tf_zs_skills_ego_abstarget_{source}_to_{target}_seed{seed}_mods.json`.
The thesis data reader now accepts this transfer arm; all seven pipeline unit
tests pass. Import provenance is `ABS_TARGET_TRANSFER_2026-11-11.json` in the
snapshot. Full report: `tmp/abs_target_transfer_20261111/RESULTS.md`.

**Remaining J transfers complete (2026-11-12).** All 70 additional base-only
jobs / 630 episodes passed, without retraining or rerunning primary-game pairs.
Full J transfer coverage is now 16 pairs / 80 jobs / 720 episodes, imported into
`frozen_2026-11-07/` with `ABS_TARGET_TRANSFER_ADDITIONAL_2026-11-11.json`.
J exceeds untrained skills in 9/16 pairs and fixed rules in 0/16. J exceeds SCOPE
in NTG → Breakout (1.33 vs 1.20) and NTG → Freeway (1.93 vs 0), with large seed spread.
All existing numbers and main transfer figures are unchanged. Full report:
`tmp/abs_target_transfer_additional_20261111/RESULTS.md`; thesis-facing report:
`masterthesis/notes/J_transfer_results_2026-11-12.md`. No text recommendation is adopted automatically.
