# Thesis release notes

The frozen result snapshot is the reference. The evaluation suite records the
902 inputs consumed by the thesis figure manifest and their SHA-256 hashes. 887
come from `frozen_2026-11-07` and 15 from the Seaquest rescue-metric rerun of
2026-11-12. The suite also lists 5,227 replayable conditions. Unvalidated label swap results are retained in the
frozen archive but omitted from replay. Missing comparisons remain listed in the
suite. The original snapshot and its provenance files are not rewritten.

The canonical suite uses the resolved evaluation configurations available on the
two evaluation clusters and the final rerun metadata where applicable. SCOPE's
heuristic-controlled evaluation now forces the ego NOOP action at the trailing
NOOP target, matching the final reruns. Learned selectors retain their original
behavior. The suite uses one process per checkpoint and condition because
in-process evaluation order affected some historical runs.

The cluster evaluation used Python 3.12.3, JAX/jaxlib 0.10.0 with CUDA 12,
Flax 0.12.7, and NumPy 2.4.4. It loaded the source tree via `PYTHONPATH` and also
had an editable install from `pyproject.toml`. `requirements-eval.txt` captures
the installed versions as constraints. CPU and other GPU configurations may
produce different trajectories; the replay comparison reports differences rather
than treating cross-platform agreement as guaranteed.

The absolute-target ablation (J) was appended to the snapshot on 2026-11-11
(10 home-game checkpoints, 70 home evaluations, 80 zero-shot transfers).
`scripts/extend_evaluation_suite.py` added these runs and the rescue hashes to the
suite built on 2026-11-10. The J checkpoints and their configs ship inside the
snapshot. The suite marks their jobs `checkpoint_in_archive`.

The rescue-metric rerun measured successful Seaquest rescues for NEXUS symbolic
pixel, HRL pixel and HRL OC (five checkpoints, base and no enemies, nine episodes
each). The thesis takes only the rescue counters from it and keeps the frozen
returns. Several rerun return vectors differ from the frozen ones despite verified
checkpoint hashes and seeds. `checkpoint_metrics.csv` in the rerun folder lists
the comparison.

The GitHub Release asset `scope-thesis-results-v1.0.0.zip` contains the
unchanged snapshot, the rescue rerun results and provenance, and `SHA256SUMS`. The archive
verification and figure modes require no checkpoints. For policy replay also
supply the checkpoint `outputs/` tree. Keep matching `*_config.yaml`, batch
statistics and meta checkpoint files alongside the weights. The inventory found
all 330 recorded checkpoint paths across the two clusters, roughly 2.27 GB before
additional sidecars. Organize checkpoint assets in independently extractable
parts if necessary; both should extract into the same checkpoint root. The local
`docs/release/checkpoint-inventory.json` records source availability, not weight
checksums. Record checksums when assembling the release assets.

`symbolic_options` is an optional external dependency for NEXUS and HRL policy
replay. SCOPE does not distribute that dependency. Use the implementation and
configuration compatible with the thesis checkpoints. The original cluster
integration used a modified copy, so stock upstream equivalence has not been
established. `third_party/patches/symbolic_options_thesis.patch` records the
local modifications at the pinned submodule commit, including the pixel channel
stacking and optional batch statistics. It was regenerated from the local
submodule on 2026-11-12. The previous version omitted the
`hierarchical_pqn_jaxtari.py` changes. Byte equality with the cluster copies was
not checked again. The archived NEXUS/HRL results regenerate without the dependency.

The analysis generates machine-readable checkpoint means, summary and control
CSV files, transfer and robustness tables, and SVG/PDF figures. Checkpoint means
weight each training seed equally. The raw evaluation JSON preserves the episode
vectors and environment counters. `verify.py` independently recomputes the
reported summaries and checks provenance and figure assets.

The thesis should cite the actual GitHub release tag and submitted source
version. A DOI is not required for this release plan.
