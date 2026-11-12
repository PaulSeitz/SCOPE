# JAXtari environment provenance

SCOPE uses [upstream JAXtari](https://github.com/k4ntz/JAXAtari) at commit
`ae1d7a0a2146f09bd6bdbb1bebf21ce4f3e9dd94`. `evaluation-compat.patch` records
source differences from the thesis environment that matter for the used games:
Seaquest background rendering, native downscale and mod asset remapping, full
Atari action indexing through mod wrappers, and observation normalization dtype.
The installer applies the patch to a temporary upstream checkout before building.

The evaluated controller modules live in `src/concept_based_learning/jaxtari_mods`.
The runtime hook registers them by game in JAXtari's mod registry. It applies to
SCOPE, PQN, and the optional NEXUS adapter. The modules retain the upstream MIT
license and attribution in `LICENSE-JAXTARI` in the same directory.

Use JAXtari's `install-sprites` command for assets. The evaluated cluster used the
packed Seaquest background frames; the local screenshot-based renderer input was
not present on that cluster. The paper's saved environment illustrations are
included separately for regenerating figures from archived data.
