# Scripts

- `reproduce.py` verifies the archive, regenerates figures and tables, and replays checkpoints.
- `install_environment.py` installs the pinned evaluation environment.
- `run_nexus.py` runs the optional NEXUS and HRL baselines.
- `materialize_evaluation.py` rebuilds the replay specification from the historical config trees and evaluation provenance. The published workflow uses the committed specification.
- `extend_evaluation_suite.py` adds the absolute-target ablation and the rescue-metric rerun (both integrated after the 2026-11-10 suite) to the specification.
- `run_abs_target_*.py`, `run_rescue_metrics.py`, `smoke_*.py` and `status_abs_target_campaign.py` are the cluster campaign scripts that produced those results. They are kept as provenance.
