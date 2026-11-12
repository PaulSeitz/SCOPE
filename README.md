# SCOPE

SCOPE is a hierarchical reinforcement learning agent that selects a target object and routes control to a skill that acts on target-relative information. This repository contains the implementation, baselines, and reproduction tools for the thesis evaluation in JAXtari.

## Reproduce the thesis results

The thesis cites release `v1.0.0`, the state submitted with the thesis. Download `scope-thesis-results-v1.0.0.zip` from that GitHub Release and extract it. It contains the frozen snapshot `frozen_2026-11-07/` and the Seaquest rescue-metric rerun `rescue_metrics_20261112/`. Keep both folders side by side. The archive is read-only input to the commands below.

```sh
# Check all 902 inputs against the thesis hashes (Python standard library only).
python scripts/reproduce.py --reference /path/to/scope-thesis-results/frozen_2026-11-07

# Regenerate numerical tables and evaluation figures (requires Inkscape on PATH).
python -m pip install -r analysis/requirements.txt
python scripts/reproduce.py --stage figures \
  --reference /path/to/scope-thesis-results/frozen_2026-11-07 --output scope-figures

# List the workload without loading checkpoints or starting a run.
python scripts/reproduce.py --stage evaluate --dry-run
```

`analysis/requirements.txt` pins the versions that produced the thesis figures (NumPy 1.26.4, Matplotlib 3.10.5, with Inkscape 1.2.2). Install them in their own virtual environment, since the evaluation environment below pins newer versions.

For checkpoint replay, use Python 3.12 and install the pinned evaluation environment in a fresh virtual environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python scripts/install_environment.py --cuda12  # omit --cuda12 for CPU
install-sprites  # follow JAXtari's asset setup prompts
```

The installer installs SCOPE in editable mode because configurations are read from `configs/` in the checkout, so keep the checkout in place.

The installer uses upstream JAXtari at an exact commit with a small compatibility patch. SCOPE registers the evaluated mod controllers from its own package. See [environment provenance](third_party/jaxtari/README.md). There is no fork checkout requirement.

Checkpoints for the absolute-target ablation (J) ship inside the results archive, so their replay needs no separate download:

```sh
python scripts/reproduce.py --stage evaluate \
  --reference /path/to/scope-thesis-results/frozen_2026-11-07 \
  --match 'tf_seaquest_skills_ego_abstarget_seed44_eval_mods.json' --condition base \
  --output scope-replay-j
```

All other checkpoints are not part of the release. Replaying them needs a checkpoint tree that keeps the paths and configuration sidecars under `outputs/`. Run a subset first and use a separate directory for new results:

```sh
python scripts/reproduce.py --stage evaluate \
  --reference /path/to/scope-thesis-results/frozen_2026-11-07 --checkpoint-root /path/to/checkpoint-bundle \
  --match 'tf_seaquest_skills_ego_seed44_eval_mods.json' --condition base \
  --output scope-replay
```

Omit `--match` and `--condition` for the full suite. Use `--resume` to continue a replay with the same output directory. Each condition runs in a fresh process. The command compares episode return vectors with the archive and writes `comparison.json`. Replayed returns can differ from the archive on other hardware or software versions. The archive was produced with JAX 0.10.0 on CUDA 12 GPUs. Full replay is substantially more expensive than regenerating figures. `--dry-run` prints the exact workload for the selected subset.

The SCOPE and PQN paths need no NEXUS installation. To replay NEXUS/HRL, check out the `third_party/symbolic_options` submodule (`git submodule update --init`), apply `third_party/patches/symbolic_options_thesis.patch` inside it, and pass it with `--nexus-root`, or pass `--exclude-external` to omit those checkpoints. Figures from the frozen archive include all baselines without running them. See [release notes](docs/release/reproducibility.md) for baseline, data, and validation details.

## Training

The training recipes for the evaluated checkpoints are Hydra experiments under `configs/experiment/`. For example, one absolute-target ablation run:

```sh
python main.py experiment=paper_eval/abs_target_ablation/tf_seaquest_skills_ego_abstarget_seed44 WANDB_MODE=disabled
```

`configs/config.yaml` logs to the author's Weights & Biases project by default. Set `WANDB_MODE=disabled`, or set `ENTITY` and `PROJECT` to your own. A 50M-step run needs a GPU.

## Repository layout

- `src/concept_based_learning/`: SCOPE, environment adapters, and baselines. The original Python namespace is retained for compatibility.
- `configs/reproduce/thesis.json`: the resolved thesis evaluation suite and reference hashes.
- `scripts/reproduce.py`: archive verification, numerical analysis, and checkpoint replay.
- `analysis/`: the thesis evaluation figure and table generators.
- `configs/experiment/`: training recipes for the evaluated checkpoints, baseline overlays, and configs used by the tests.
- `docs/release/`: release notes, checkpoint inventory, and the cluster software environment. `docs/development/` keeps historical plans and run logs.
- `third_party/patches/symbolic_options_thesis.patch`: local changes to the optional NEXUS/HRL baseline at the pinned submodule commit.

## Citation

If you use this code, please cite the thesis (see `CITATION.cff`).

## License

SCOPE is released under the MIT License (`LICENSE`). The evaluated JAXtari mod controllers in `src/concept_based_learning/jaxtari_mods/` keep their upstream MIT license (`LICENSE-JAXTARI`). `third_party/patches/symbolic_options_thesis.patch` modifies the NEXUS implementation (`symbolic_options`) and is shared with its author's permission.
