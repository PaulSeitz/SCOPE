# Thesis-final results

The current figure generator uses `tmp/thesis_final_results/frozen_2026-11-07/` (the corrected November 7 snapshot). The older reward-fix wave is `tmp/thesis_final_results/frozen_2026-10-30/` (pulled 2026-10-30). Read each snapshot’s `README.md`; plot from its `eval/`. Checkpoints are under `checkpoints/cluster1` and `checkpoints/cluster2`, listed in `MANIFEST.tsv`. The Have list below is the older local set, not this wave. Append a finished eval into the frozen folder when its JSON is complete; do not plot a stem that is only on the cluster.

**Absolute target (2026-11-11, complete and imported).** Ten `skills_ego_abstarget` checkpoints (Seaquest/NTG, seeds 44–48), all 70 home-condition evaluations and 630 episodes are validated in the current `frozen_2026-11-07/` basis. New checkpoints are under `checkpoints/{cluster1,cluster2}/abs_target_ablation/`. Import hashes/provenance: `ABS_TARGET_2026-11-11.json`. Full comparison: `tmp/thesis_final_results/abs_target_2026-11-11/RESULTS.md`. No transfer evaluations. Existing score files were not overwritten.


**Downscaling (2026-11-04).** Pixel HRL and pixel NEXUS do not use native downscaling. The runs that do are pixel-outline skills and the pixel-slot tandem. Those are being retrained with it off, seeds 44–48, both homes, then the evals that load the new checkpoints. Stems end in `_nodown`. Seed 44 was already running (`logs/probe_gpu{1,2}.log`). Seeds 45–48 and the transfer evals are queued on cluster1 (`logs/nodown_gpu{1,2,3}.log`). Frozen checkpoints are not overwritten.

**Sticky actions stay 0.** One Seaquest seed with probability 0.25, eval still sticky-off: pixel PQN base 19289 against 15803, ego 22819 against 15208. Not a different regime, and a full sticky rerun is not worth it. JAXtari and CleanRL also use 0. The eight Seaquest → target transfers of the sticky ego checkpoint (`tf_zs_skills_ego_seaquest_to_*_seed44_sticky`) run on cluster1 GPU 3 (`logs/sticky_ego_transfers.log`), then that GPU continues its nodown queue. Appendix placeholder `sec:appendix-sticky-actions`; thesis note `masterthesis/notes/new_runs_2026-11-04.md`.

**Tandem skills under the home heuristic (2026-11-03).** Same transfer eval as the cross-game skill transfers, but the destination is the source game. No learned selector is loaded. Both arms are in `tmp/thesis_final_results/frozen_2026-10-30/eval/`: `tf_zs_tandem_{ocnative,pixel_slots}_skills_{seaquest,namethisgame}_to_{seaquest,namethisgame}_seed{44-48}`. The pixel-slot runs needed an empty card; the first attempt on a shared cluster2 card ran out of memory.

**Name This Game HRL evals (2026-11-03).** Pixel seeds 44–47 and object-centric seeds 44–48 are in the frozen eval folder. Pixel seed 48's eval was restarted on cluster1 GPU 0 (`logs/ntg_hrl_pixel_seed48.log`); the checkpoint was already saved. Object-centric seeds 45 and 47 have no separate `no_shark` mod; the thesis uses `no_shark_short_tentacles`, which is present. Background-only re-evals for the home-heuristic tandem skills (object-centric and pixel-slot, Name This Game, seeds 44–48) are on cluster1 GPU 1 (`logs/ntg_heu_bg.log`) and get spliced into the full files when they finish. GPU 2 stays empty.

**Done and in `tmp/thesis_final_results/frozen_2026-10-30/eval/`.** The frozen reward-fix wave, including Seaquest HRL pixel 47–48, NEXUS symbolic pixel 48, neural object-centric 47–48, neurosymbolic object-centric 44–48 (the six re-run mods are spliced into seeds 45, 47, and 48), symbolic object-centric 44–48, the 20 home-heuristic tandem evals (object-centric and pixel-slot), and Name This Game object-centric HRL seeds 44, 46, and 48.

**Name This Game HRL (2026-11-02, queued).** Corrected 50M recipe, seeds 44–48, same as Seaquest: pixel CNN on cluster1 GPU 1 (`cbl_namethisgame_hrl_pixel_optim50m`), object MLP on cluster2 GPUs 1–3 beside ollama (`cbl_namethisgame_hrl_oc_optim50m`). The October 23 object baseline stays the pre-repair run.

**Name This Game `no_shark` (2026-11-01).** Shark parked off-screen, tentacles left on the normal update, observation marks the shark inactive. Not in `paper_core`. Eval-only on the five ego checkpoints and the five pixel-PQN checkpoints. No corrected Name This Game pixel HRL exists, so that arm is not included. Spliced into the full evals when the queue finishes.

**Pulled 2026-11-01** into `tmp/thesis_final_results/frozen_2026-10-30/eval/`, including the spliced Name This Game `change_background_color` key. Beam Rider pixel seeds 47 and 48 finished after that pull and are not in the folder yet.

**Transfer PQN and Name This Game background (2026-10-31).** Pixel PQN (`tf_{env}_pqn_pixstack`) on cluster1 GPUs 0, 1, and 3. Native OC PQN (`tf_{env}_pqn_ocnative`, 2×128) on cluster2 GPUs 1, 2, and 3 beside ollama. Seven transfer games, seeds 44–48, home eval only. GPU 2 on cluster1 stays empty. Name This Game `change_background_color` re-evals (one mod, 9 episodes) run on cluster1 GPU 0 and cluster2 GPU 2 before those cards start their PQN queues, then `scripts/merge_ntg_bg_eval.py` splices the key into the full eval JSON/CSV. The mod is not in `paper_core`.

**Native ceilings and floors (2026-10-30, queued on cluster1).** Same corrected ego recipe as the frozen homes, seeds 44–48, home eval only: `tf_{spaceinvaders,beamrider,asteroids,pong,breakout,freeway,phoenix}_skills_ego_seed{44-48}`. Floors rerun for all nine suite games (`random_actions`, `heuristic_random`, `fixed_ego_rules`; `constant_ego` on Seaquest and Name This Game only), 9 eval seeds × 3 episodes. GPU 2 stays empty. Logs `logs/c1_native_ceil_gpu{0,1,3}.log`. Old floor JSONs on cluster1 were moved to `outputs/models/paper_eval/eval/_pre_floor_rerun_2026-10-30/` so `--resume` does not keep them. Not in the frozen folder until the new files are pulled.

Two lists. **Pipeline** is what still has to run or be pulled. **Have** is the older local set.

Edit Pipeline by adding a row. Status is `queued`, `running`, `blocked`, or `done-not-pulled`. `blocked` means the configs exist, but do not start the row until the gate named in it has been decided.

- A re-run: delete the old local and cluster files, remove the stem from Have, add a `queued` row.
- A finished run that is not local yet: `done-not-pulled`, and list the stems in the row.
- After those files are in `tmp/thesis_final_results/eval/` and the scores are real: add the stems under the matching Have heading and delete the row.

For a stem `S` the files are `S_mods.json`, `S_mods_summary.csv`, and `S_mods_episodes.csv`.

- Local figure basis: `tmp/thesis_final_results/frozen_2026-10-30/eval/`
- Older local Have files: `tmp/thesis_final_results/eval/`
- Cluster eval: `~/ConceptBasedLearning/outputs/models/paper_eval/eval/` on cluster1 (`pseitz@labmeles`)
- Cluster checkpoint: `~/ConceptBasedLearning/outputs/models/paper_eval/thesis_final/<train>/`, where `<train>` is `S` without a trailing `_eval`. Floors have no checkpoint.

Use the summary CSV for per-mod means. Do not pool scores across games. `oc_flat_rel` is the relative object list plus `is_target`. It is not the full-context OC test (`oc_flat_rel_role`).

Main environments are Seaquest and Name This Game. Time Pilot is not a main environment. The final suite, once this seed-44 wave is accepted, uses five seeds: 44, 45, 46, 47, and 48. Do not restart the jobs already running. Active local evals are attribution floors plus Seaquest NEXUS HRL. Flat PQN and every CBL skill result were archived on 2026-10-24.

## Pipeline

**Restart (2026-10-24, 17:18), two GPUs only.** GPU 2 stays with the other project's BlendeRL jobs. GPU 3 is unused on purpose. Time Pilot is dropped as a home environment and is not a transfer target in this wave.

The first pair is the corrected Seaquest seed-44 skill learner, so the cross-skill choice can be checked before anything else is committed:

| GPU | first | then, in order | log |
| --- | --- | --- | --- |
| 0 | `tf_seaquest_skills_ego_seed44` (`OFF_POLICY: true`) | home mod eval, then transfers to Name This Game, Space Invaders, Asteroids, Pong, Breakout, Freeway, Phoenix, Beamrider, then NEXUS symbolic seed 44 | `logs/ego_seed44_then_nexus44_gpu0.log` |
| 1 | `tf_seaquest_skills_ego_no_cross_skill_seed44` (`OFF_POLICY: false`) | the same eval and the same eight transfers, then NEXUS symbolic seeds 45 and 46 | `logs/ego_nocross_seed44_then_nexus_gpu1.log` |

Both use the corrected λ-return, half-MSE, and learning rate 2.5e-4. The cluster has `training/returns.py` and the updated `train_skills.py`. Flat PQN and tandem use that same return, so they are retrains too; their trainers (`flat_pqn_train.py`, `train_tandem.py`) and the eval harness were copied to the cluster on 2026-10-24 after these two jobs had already started. The running processes keep the trainer they imported. Stop the chain after the Seaquest evals if that comparison changes the plan. NEXUS starts only after that GPU's transfers, so it can be cut without touching the comparison.

**AVOID separation (2026-10-26, 18:45).** The reference skill reward no longer treats the first step away from the target as a +1 terminal. Distance increase is +0.01 and the return continues. Contact, configured kill events, and life loss stay -1 and terminal. `materialize_thesis_final.py` now writes `AVOID_SPARSE_ONLY: false`. Replacement runs use the stem token `avsep`. Old-reward Seaquest skill evals and checkpoints are in `tmp/thesis_final_results/archive/pre_avoid_separation_2026-10-26/`.

**Contact events (2026-10-26, evening).** The first Seaquest `avsep` job was stopped after about 16 minutes and its checkpoint deleted. It was learning three bad signals: a diver leaving the screen counted as collection, a sprite cleared to (0, 0) could collide with a player sitting there, and destroying the target was still an AVOID failure. Those are fixed in `skill_rewards.py`. Freeway overlap now uses bottom-edge y. COLLECT still scores +1 for nonlethal contact with whatever object was selected, including an enemy.

The repaired module and the train config (`AVOID_SPARSE_ONLY: false`, `AVOID_KILLED_EVENT_MODE: none`) were copied to cluster1 and the same ten-run chain was started again at 19:15 (`logs/avsep_ego_seed44_gpu0.log`, wandb `utawhpfm`). Other skill families stay off until this seed shows it still learns.

Cluster2 still has no free GPU. On cluster1, GPU 3 stays free, GPU 2 is the other project, and GPU 1 is still the flat PQN seeds. That baseline does not use the skill reward.

Cross-skill stays the main recipe. The skill chains that were queued behind GPU 0 (`logs/gpu0_extra_seeds.log`) were already killed with the old-reward restart and must not be resumed. GPU 1 continues pixel PQN and OC PQN seeds 45–48 with home evals. No extra transfers in this extension. Time Pilot stays out.

NEXUS HRL checkpoints already on disk were trained before the λ-return repair. The cluster trainer did not contain `lambda_return_targets` until 2026-10-24, and the HRL evals finished on 2026-10-23. Do not reuse them as corrected results. The symbolic retrain below uses the repaired trainer.

Archived 2026-10-24 to `tmp/thesis_final_results/archive/pre_bootstrap_rerun_2026-10-24/`. Those eval files and their `thesis_final` checkpoints were deleted locally and on cluster1. The set is every `tf_` result trained with the ConceptBasedLearning λ-return: skill arms, tandem, transfers, no-cross-skill, reward-semantics v2, and flat PQN (`pqn_pixel`, `pqn_oc`), which uses the same bootstrap. Floors and NEXUS HRL stay. Historical non-`tf_` files on the cluster were not touched. The mixed `all_mod_eval_*.csv` dumps were archived and removed from the active eval directory.

| status | id | run |
| --- | --- | --- |
| done | action-frame-sensitivity | Seaquest seeds 44–48, native-OC and nodown pixel skills with `EGO_ACTION_FRAME: screen`, train + home eval. Finished 2026-11-05 ~06:06 on cluster2, in the frozen `eval/` folder. Appendix check only. |
| done | abspos-intervention | Eval-only on the 10 `skills_ego_abspos` checkpoints: control, frozen, mirror_x. In frozen `eval/abspos_intervention/`. Control reproduces shifted layouts in 5/5 seeds but base in 4/5 (Seaquest) and Seaquest random_spawns in 3/5; cause not traced. |
| done | prefix-native-oc | `tf_{env}_skills_ego_ocnative_seed{44-48}`: reference prefix + full native-OC input. Finished 2026-11-05 on cluster2, evals in frozen `eval/`. Evals ran one mod at a time (`CBL_EVAL_MOD_WORKER=1`). |
| running | scene-rows | `tf_{env}_skills_ego_scenerows_seed{44-48}`: relative scene without the reference prefix, target marked only by `is_target`. Started 2026-11-05 13:47 on cluster2, seeds 44–45 first (GPU 3 only for Seaquest seed 44), then 46–48 behind the flat reruns. Logs `logs/rows_gpu*.log`. |
| queued | ntg-nodown-bg-reeval | 10 `tf_namethisgame_{skills_pixel_outline,tandem_pixel_slots}_seed*_nodown_eval_bg`. Configs local only, not on cluster1, not started. Pixel-outline seed 48 needs its checkpoint first. Results need merging into the main eval files. |
| blocked | flat-pqn-scene | `tf_{env}_pqn_ego_scenerel_seed{44-48}` are **invalid; rerun after the fix**. `make_flat_oc_build_env` stacks on the semantic env, which has no auto-reset; the hierarchical trainer resets it itself, but `flat_pqn_train.py` relies on the wrapper to. After the first `done` the env reports `done` on 93.5% (Seaquest) / 100% (Name This Game) of steps, so training sees almost no reward. Their files in the frozen `eval/` folder must not be plotted. No frozen checkpoint has `OC_INPUT_MODE` in its saved config, so thesis results are not affected. The earlier flat `oc_simple_*` / `oc_embed` probes (e.g. oc-format-probes) used this path. |
| running | nodown-rerun | cluster1, 175 evals complete and pulled to frozen `eval/` on 2026-11-05 13:15. Still running: Name This Game pixel-outline seed 48 (train, home eval, 8 transfers, GPU 1) and 22 pixel-slot tandem transfers (GPU 2). Name This Game nodown evals have no `change_background_color` or `no_shark` keys, unlike the frozen files. |
| done | skill-context-ablation | Two RQ1 attribution arms, both homes, seeds 44–48, home eval only, cluster2 GPUs 1, 2, 4–7, finished 2026-11-05 01:48. All 20 evals are in the frozen `eval/` folder (`tf_{env}_skills_ego_{scenerel,abspos}_seed*_eval`); checkpoints stay on cluster2. Seaquest scene evals failed on GPU 1 (seeds 44, 47) with 9 parallel workers at 0.078 and passed when rerun on GPUs with more free memory; `EVAL_PARALLEL_MODS: false` in their YAMLs has no effect. cluster2's JAXAtari fork and mod list gained `no_shark` during this run; Name This Game seeds 44–47 were re-evaluated. Results: `masterthesis/notes/context_ablation_2026-11-04.md` §6. |
| queued | pixel-meta-raw-final | Author selected raw-image pixel_slots after the successful Seaquest44 pilot. Recommend Seaquest45/46 and Name This Game44/45/46, full home evals and skill transfers; reuse pilot44 only after saved-config checks. Isolated configs/chain runner prepared, no jobs launched. Orders: `docs/pixel_meta_finalization_2026-10-27.md`. |
| done-not-pulled | pixel-meta-slot-pilot | Seaquest seed 44 base-only pilot, pulled locally. Saved config matches the final `pixel_slots` recipe, so the checkpoint is reused for the full home eval and skill transfers (`ps_seaquest_slots_seed44_full_eval`). Base return 25211±7118 is one seed's episode spread, not a final multi-seed result. |
| dropped | pixel-meta-candidate-pilot | Dropped with the hybrid pixel selector. The accepted selector is raw-image `pixel_slots`. Do not launch the candidate-marked fallback. |
| running | probe-a-seeds-45-46 | GPU 0 seed 45, GPU 2 seed 46. Same early-collect centre-ray recipe as probe A seed 44. Logs `logs/probe_a_early_seed45_gpu0.log` and `logs/probe_a_early_seed46_gpu2.log`. |
| dropped | thesis-final-A-recipe | Stopped 2026-10-28 ~02:10. Those skill runs trained with NAVIGATE folded into COLLECT and JUMP_OVER folded into AVOID. Seaquest ego checkpoints for seeds 44–46 were deleted. |
| running | rewardfix-reference | Corrected three-head rewards. Seaquest ego seeds 44–46 on cluster1 GPUs 0, 1, 3; seeds 47–48 on cluster2 GPUs 0, 1. Each seed then does the home mod eval and eight transfers, then the matching Name This Game ego chain. Cluster2 GPU 2 is the corrected channel-stacked pixel PQN, Seaquest then Name This Game, seeds 44–48. Cluster2 GPU 3 is the Name This Game half of that pixel PQN. GPU 2 on cluster1 stays empty. Logs `rewardfix_gpu*.log`. |
| running | oc-pqn-3x128 | Extra monolithic OC PQN, MLP 3×128, not a replacement. The completed 2×128 native OC PQN (`tf_{seaquest,namethisgame}_pqn_ocnative_seed44-48`) stays the main result. Both homes, seeds 44–48, home mod eval only, no cross-game transfer. Stems `tf_{env}_pqn_oc_h128l3_seed{44-48}`. Cluster2 GPUs 0–3, cap 12% beside ollama. Logs `oc_h128l3_gpu*.log`. |
| done | rewardfix-wave | Frozen corrected rewards. Follow-on queues `logs/wave_c*.log` start when the reference bash on that GPU exits. Screen, no-POV, and pixel-outline, both homes, seeds 44–48, with home evals and transfers. Pixel-slot tandem retrained the same way (old seeds 44–46 archived). Aligned OC tandem (`oc_native`, four-frame float32) both homes, home evals and ego-skill transfers. Native-OC skills, home evals only. Seaquest acting-skill-only (`OFF_POLICY: false`), home evals only. Float32 raw-OC PQN both homes, home evals only, no cross-game OC transfer. Seaquest NEXUS neural/neurosymbolic/symbolic × object and pixel, seeds 44–48, plus HRL object and pixel. Cluster2 pixel NEXUS/HRL cap is 30% with preallocate off; the MLP queue cap is 10%. |
| done | pixel-meta-final | Cluster2, started 2026-10-27 ~17:35, memory cap 0.40 beside ollama. GPU 0 Seaquest `pixel_slots` seed 45 then Name This Game 45. GPU 1 Seaquest 46 then Name This Game 46. GPU 2 reuses the Seaquest 44 pilot for the full home eval and eight skill transfers. GPU 3 Name This Game 44. Candidate-marked and hybrid-pixel stay off. Logs `logs/pm_final_*`. |
| done-not-pulled | oc-format-probes | Finished 2026-10-27 ~14:58 on cluster2. Flat PQN, but not raw OC: `oc_simple_rel_role` / `oc_simple_role` / `oc_embed`, 512 envs, seed 44. Base means 80 / 196 / 133. Rescues, collects, and surface events are 0. JSONs on cluster2, not pulled. |
| done-not-pulled | oc-raw-pilot | Monolithic raw-OC PQN, Seaquest seed 44. Base return 698±22. No rescues, no surfacing. This is the conventional raw-OC baseline, not an OC tandem result. JSON pulled locally. |
| dropped | oc-specialized-formats | Deleted the finished specialized OC probes and the `oc_flat_rel` skill checkpoints. Unstarted `oc_flat_rel`, rel-role `tandem_oc`, and hybrid `tandem_pixel` stems were removed from the cluster1 queues. Hybrid pixel and the 512-wide MLP are dropped, not deferred. |
| superseded | oc-raw-tandem | Replaced by the aligned `ocnative` tandem in the frozen wave. The latest-frame raw tandem is not a result. |
| done | nexus-hrl-input-matrix | Included in `rewardfix-wave`. Seaquest seeds 44–48 are in the frozen folder except six neurosymbolic object-centric mods still rerunning (see the top of this log). Do not launch a second copy. |
| done | ntg-match-seaquest | Cluster2 GPU 2, after the Seaquest slot-reuse chain finished. Name This Game raw-OC PQN seed 44 plus its home mod eval, then pixel PQN seeds 44–48 with home evals. Same recipes as the Seaquest runs. Skill arms, pixel-slot tandem, and raw-OC tandem for Name This Game were already queued. No NEXUS. Log `logs/ntg_match_gpu2.log`. |
| done | flat-pqn-extra-seeds | GPU 1 only. Pixel PQN and OC PQN seeds through 48. Do not restart. |
| superseded | seaquest-floors-pre-early | The Seaquest floor JSONs in Have used the selector from before early collect. The refresh overwrites those three stems. Other games' floors stay. |
| dropped | timepilot-and-stale-yamls | Time Pilot is not in the regenerated suite. Old no-cross and pre-repair skill YAMLs were removed from `configs/experiment/paper_eval/thesis_final/` when the suite was regenerated. Probe and avsep YAMLs were kept. |
| blocked | reward-semantics-v2 | Archived and removed from the active eval dirs. Rejected reward contract: base 11536 vs old ego 23326. Do not propagate the flags. |
| blocked | screen-frame | Cancelled 2026-10-24 before a checkpoint was written. Partial save dirs deleted. Do not resume. Waiting on the bootstrap bug note. |
| blocked | no-cross-skill-seed44 | Archived and removed. Seaquest base 8049 vs ego seed 44 base 23326. Keep `OFF_POLICY: true` when skill jobs restart. |
| blocked | screen-transfer | 27 zero-shot evals `tf_zs_skills_screen_seaquest_to_{asteroids,beamrider,breakout,freeway,namethisgame,phoenix,pong,spaceinvaders,timepilot}_seed{44,45,46}`. Start after the three screen checkpoints demonstrate nontrivial Seaquest competence. |
| blocked | screen-floors-ceilings | Matched screen-specific heuristic and fixed-rule floors, plus native screen ceilings. Not materialized. Separate from `screen-frame`. |
| blocked | oc-flat-rel-role | Never started. It was chained after screen seed 44 and the chain was killed. Waiting on the bootstrap bug note. |
| blocked | extended-transfer-pixel | All 54 finished, then archived and removed with the other CBL-bootstrap checkpoints. |
| done-not-pulled | nexus-symbolic | Seeds 44/45/46 finished. Base means 413 / 262 / 247. JSONs on cluster1, not in Have. |
| blocked | extended-transfer-fast | All 48 finished, then archived and removed. The no-POV slice used target-aligned actions, so it is not the clean screen-frame comparison. |
| blocked | hrl-ntg | Name This Game HRL seeds 44/45/46 are on cluster1, base means 9447 / 8071 / 8318. Same provenance problem as the other HRL checkpoints: trained before the λ-return repair. Not corrected results. |
| blocked | hrl-timepilot | Time Pilot is not a main environment. Seeds 44 and 45 are still in Have, and seed 46 is on cluster1, but those checkpoints predate the NEXUS λ-return repair. Do not treat them as corrected results. |
| done-not-pulled | beamrider-floors | Floor JSONs stay on cluster1 and are not in the local pull: `tf_floor_random_actions_beamrider`, `tf_floor_heuristic_random_beamrider`, `tf_floor_fixed_ego_rules_beamrider`. The Beamrider skill train and Seaquest-to-Beamrider transfer were archived and removed. |

**Cross-skill decision (2026-10-24):** the archived no-cross-skill run, trained with the broken bootstrap, scored Seaquest base 8049 vs 23326. That comparison is being repeated on the corrected return. Do not treat the archived gap as the decision.

**Reward gate decision (2026-10-23):** retain the thesis-final reward contract and
cancel the conditional full reward rerun. The combined v2 diagnostic is a
reward-design failure rather than a loading/calculation bug. The source-checked
postmortem is in
`Documents/latex_projects/masterthesis/notes/reward_semantics_v2_postmortem_2026-10-23.md`.
Do not start another reward smoke unless an advisor explicitly needs the
AVOID-only causal isolation; it would not be a candidate for automatic adoption.

## Have

35 stems. Flat PQN and CBL skill results are not here.

Archive: `tmp/thesis_final_results/archive/pre_bootstrap_rerun_2026-10-24/`.

## Attribution floors — random actions

- `tf_floor_random_actions_seaquest`
- `tf_floor_random_actions_namethisgame`
- `tf_floor_random_actions_timepilot`
- `tf_floor_random_actions_spaceinvaders`
- `tf_floor_random_actions_asteroids`
- `tf_floor_random_actions_pong`
- `tf_floor_random_actions_breakout`
- `tf_floor_random_actions_freeway`
- `tf_floor_random_actions_phoenix`

## Attribution floors — heuristic + random skills

- `tf_floor_heuristic_random_seaquest`
- `tf_floor_heuristic_random_namethisgame`
- `tf_floor_heuristic_random_timepilot`
- `tf_floor_heuristic_random_spaceinvaders`
- `tf_floor_heuristic_random_asteroids`
- `tf_floor_heuristic_random_pong`
- `tf_floor_heuristic_random_breakout`
- `tf_floor_heuristic_random_freeway`
- `tf_floor_heuristic_random_phoenix`

## Attribution floors — heuristic + fixed ego rules

- `tf_floor_fixed_ego_rules_seaquest`
- `tf_floor_fixed_ego_rules_namethisgame`
- `tf_floor_fixed_ego_rules_timepilot`
- `tf_floor_fixed_ego_rules_spaceinvaders`
- `tf_floor_fixed_ego_rules_asteroids`
- `tf_floor_fixed_ego_rules_pong`
- `tf_floor_fixed_ego_rules_breakout`
- `tf_floor_fixed_ego_rules_freeway`
- `tf_floor_fixed_ego_rules_phoenix`

## Attribution floors — constant ego

- `tf_floor_constant_ego_seaquest`
- `tf_floor_constant_ego_namethisgame`
- `tf_floor_constant_ego_timepilot`

## NEXUS HRL still on disk, not corrected

These checkpoints were trained before the NEXUS λ-return repair. Seaquest is a main environment. Time Pilot is not.

- `tf_seaquest_nexus_hrl_seed44_eval`
- `tf_seaquest_nexus_hrl_seed45_eval`
- `tf_seaquest_nexus_hrl_seed46_eval`
- `tf_timepilot_nexus_hrl_seed44_eval`
- `tf_timepilot_nexus_hrl_seed45_eval`

Name This Game HRL seeds 44/45/46 and Time Pilot HRL seed 46 are still on cluster1 and were not archived. They are not in this local folder. Time Pilot floors above are reusable controls, not a main-environment result.

**Absolute-target two-way transfer (2026-11-11, complete and imported).**
Ten base-only Seaquest ↔ NTG jobs (90 episodes), all successful, now in the active
`frozen_2026-11-07/eval/` basis. SQ → NTG: 702 ± 354; NTG → SQ: 29 ± 26,
across five checkpoint means. Provenance: `ABS_TARGET_TRANSFER_2026-11-11.json`.
Report: `tmp/abs_target_transfer_20261111/RESULTS.md`. No checkpoint retraining.

**Remaining J transfers complete (2026-11-12).** All 70 additional base-only
jobs / 630 episodes passed, without retraining or rerunning primary-game pairs.
Full J transfer coverage is now 16 pairs / 80 jobs / 720 episodes, imported into
`frozen_2026-11-07/` with `ABS_TARGET_TRANSFER_ADDITIONAL_2026-11-11.json`.
J exceeds untrained skills in 9/16 pairs and fixed rules in 0/16. J exceeds SCOPE
in NTG → Breakout (1.33 vs 1.20) and NTG → Freeway (1.93 vs 0), with large seed spread.
All existing numbers and main transfer figures are unchanged. Full report:
`tmp/abs_target_transfer_additional_20261111/RESULTS.md`; thesis-facing report:
`masterthesis/notes/J_transfer_results_2026-11-12.md`. No text recommendation is adopted automatically.
