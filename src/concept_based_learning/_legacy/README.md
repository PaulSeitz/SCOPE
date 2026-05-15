# Legacy Code

These files are preserved for reference but are **not part of the active training path**.

## train_meta_agent.py

The sequential approach: train skills first (frozen), then train the meta-policy separately.
Superseded by `training/train_tandem.py`, which trains both heads jointly in a single loop.
The `meta_only` mode in `main.py` still uses this for loading pre-trained skills + running
a standalone meta-policy phase.

## affordance_pointer_network.py

An affordance-masked pointer network for System 2. Factorizes the target-selection decision
into an intent head (skill affordance) and a spatial attention head (pointer network).
Deferred: the active meta network (`networks/meta_network.py`) is a simpler MLP that picks
the current target object directly. The affordance masking step comes later.
