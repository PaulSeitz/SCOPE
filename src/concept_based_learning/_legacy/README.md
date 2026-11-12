# Legacy Code

These files are preserved for reference but are **not part of the active training path**.

## train_meta_agent.py

The sequential approach: train skills first (frozen), then train the meta-policy separately.
Superseded by `training/train_tandem.py`, which trains both heads jointly in a single loop.
The `meta_only` mode in `main.py` still uses this for loading pre-trained skills + running
a standalone meta-policy phase.
