# Code state of the thesis results (2026-11-06)

This tag records the code as it stood when the thesis results were finalized. It is not the exact commit each run used.
Runs started before this date used a mix of commits and hand-synced uncommitted files on two clusters.

- Training of the frozen results (`frozen_2026-10-30`) predates this tag. A retrain of the reference on current code
  (`skills_ego_recheck`, two seeds in both primary games) showed no directional drift, though single seeds vary strongly.
- On 2026-11-06, cluster2 matched this tree in all of `src/`. cluster1 was behind on `baselines/flat_oc_features.py`,
  `networks/skill_network.py`, `paper_eval/eval_harness.py`, `pqn_visualizer.py`, `training/train_skills.py` and lacked
  `observations/ego_context.py`. It only ran arms that do not use those additions, namely pixel skills, the reference,
  tandem pixel and HRL evaluations.
- JAXtari comes from the fork `PaulSeitz/JAXAtari`, branch `feat/seaquest-robustness-mods`, same tag name. The game
  files were identical on all machines. cluster1 had older Seaquest mods, which it did not evaluate.
- HRL and NEXUS use the submodule `third_party/symbolic_options` at its pinned commit plus
  `third_party/patches/symbolic_options_thesis.patch`, since that repository belongs to someone else.
- Checkpoints and result JSONs are not in the repository, and live in `tmp/thesis_final_results/`.
