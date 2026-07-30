"""Shared training schedule helpers."""

import optax

# Default skill epsilon-greedy schedule (skills converge quickly; meta benefits early).
DEFAULT_SKILL_EPS_START = 0.3
DEFAULT_SKILL_EPS_FINISH = 0.05
DEFAULT_SKILL_EPS_DECAY = 0.35


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


def skill_learning_rate_schedule(config, num_updates=None):
    """
    Constant ``SKILL_LR`` (fallback ``LR``), or linear decay when ``SKILL_LR_FINISH`` is set.

    ``SKILL_LR_DECAY`` uses the same semantics as ``EPS_DECAY`` (ratio of updates if ≤1).
    """
    lr_start = float(config.get("SKILL_LR", config["LR"]))
    lr_finish = config.get("SKILL_LR_FINISH", None)
    if lr_finish is None:
        return lr_start
    return optax.linear_schedule(
        lr_start,
        float(lr_finish),
        eps_decay_transition_steps(
            config.get("SKILL_LR_DECAY", 1.0),
            num_updates,
        ),
    )
