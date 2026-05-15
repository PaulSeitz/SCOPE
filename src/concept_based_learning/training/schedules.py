"""Shared training schedule helpers."""


def eps_decay_transition_steps(decay_cfg, num_updates=None) -> int:
    """
    Steps for ``optax.linear_schedule(..., transition_steps=...)``.

    Compatibility behavior:
    - ``decay_cfg <= 1`` and ``num_updates`` provided: treat as a ratio of total updates.
      Example: ``0.8`` means decay over 80% of training updates.
    - otherwise: treat ``decay_cfg`` as an explicit transition length in updates.
    """
    decay = float(decay_cfg)
    if num_updates is not None and decay <= 1.0:
        return max(1, int(round(float(num_updates) * decay)))
    return max(1, int(round(decay)))
