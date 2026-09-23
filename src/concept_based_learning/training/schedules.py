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


def num_updates_decay(config) -> int:
    """Outer-loop updates used for decay windows (matches ``flat_pqn_train``)."""
    total = int(config["TOTAL_TIMESTEPS"])
    decay_total = int(config.get("TOTAL_TIMESTEPS_DECAY", total))
    return max(1, decay_total // int(config["NUM_STEPS"]) // int(config["NUM_ENVS"]))


def optax_optimizer_transition_steps(config, num_updates) -> int:
    """
    Optax schedule length matching ``flat_pqn_train`` LR decay.

    Each ``apply_gradients`` advances the schedule once, and hierarchical PQN
    does ``NUM_EPOCHS × NUM_MINIBATCHES`` gradient steps per outer update.
    """
    return max(
        1,
        int(num_updates) * int(config["NUM_MINIBATCHES"]) * int(config["NUM_EPOCHS"]),
    )


def skill_learning_rate_schedule(config, num_updates=None):
    """
    Skill LR schedule.

    Priority:
    1. ``LR_LINEAR_DECAY: true`` → linear ``SKILL_LR`` (fallback ``LR``) → ``1e-20``
       over ``TOTAL_TIMESTEPS_DECAY`` (same optax-step counting as flat PQN).
    2. Else constant ``SKILL_LR``, or linear decay when ``SKILL_LR_FINISH`` is set
       (``SKILL_LR_DECAY`` uses the same semantics as ``EPS_DECAY``).
    """
    lr_start = float(config.get("SKILL_LR", config["LR"]))
    if config.get("LR_LINEAR_DECAY", False):
        return optax.linear_schedule(
            init_value=lr_start,
            end_value=1e-20,
            transition_steps=optax_optimizer_transition_steps(
                config, num_updates_decay(config)
            ),
        )
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


def meta_learning_rate_schedule(config, num_updates=None):
    """
    Meta LR schedule (tandem).

    Same ``LR_LINEAR_DECAY`` / ``TOTAL_TIMESTEPS_DECAY`` semantics as skills;
    otherwise constant ``META_LR`` (fallback ``0.1 * LR``), or linear decay when
    ``META_LR_FINISH`` is set.
    """
    lr_start = float(config.get("META_LR", float(config["LR"]) * 0.1))
    if config.get("LR_LINEAR_DECAY", False):
        return optax.linear_schedule(
            init_value=lr_start,
            end_value=1e-20,
            transition_steps=optax_optimizer_transition_steps(
                config, num_updates_decay(config)
            ),
        )
    lr_finish = config.get("META_LR_FINISH", None)
    if lr_finish is None:
        return lr_start
    return optax.linear_schedule(
        lr_start,
        float(lr_finish),
        eps_decay_transition_steps(
            config.get("META_LR_DECAY", 1.0),
            num_updates,
        ),
    )
