import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_COLLECT,
    LABEL_NOOP,
)
from concept_based_learning.environments.skill_rewards import (
    avoid_survival_reward_from_config,
    extract_skill_rewards,
    penalize_collect_death_from_config,
)
from concept_based_learning.observations.action_space import make_target_aligned_action_mappers


def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost, info=None, config=None):
    del info
    return extract_skill_rewards(
        prev_raw_obs,
        next_raw_obs,
        prev_target,
        next_target,
        life_lost,
        avoid_survival_reward=avoid_survival_reward_from_config(config),
        penalize_collect_death=penalize_collect_death_from_config(config),
        config=config,
    )


# Breakout: player at fixed y=189. Actions: NOOP=0, FIRE=1, RIGHT=2, LEFT=3.
# Player moves only horizontally, so vertical ego directions collapse to NOOP.
breakout_action_mapper, breakout_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=     [2, 2, 0, 3, 3, 3, 0, 2],  # R DR D DL L UL U UR
    move_fire_table=[2, 2, 1, 3, 3, 3, 1, 2],
)


class BreakoutSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # 1 NOOP + 1 FIRE + 1 ball
        super().__init__(env, max_objects=3, meta_reserved_slots=2)

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        ball = obs.ball

        ball_x = jnp.expand_dims(ball.x, axis=0) if ball.x.ndim == 0 else ball.x
        ball_y = jnp.expand_dims(ball.y, axis=0) if ball.y.ndim == 0 else ball.y
        ball_w = jnp.expand_dims(ball.width, axis=0) if ball.width.ndim == 0 else ball.width
        ball_h = jnp.expand_dims(ball.height, axis=0) if ball.height.ndim == 0 else ball.height

        # Preserve visibility supplied by the launch adapter.
        ball_active = jnp.atleast_1d(ball.active).astype(jnp.int32)
        ball_label = jnp.array([LABEL_COLLECT], dtype=jnp.int32)

        pad_size = self.game_max_objects - 1
        pad_x = jnp.zeros(pad_size, dtype=ball_x.dtype)
        pad_y = jnp.zeros(pad_size, dtype=ball_y.dtype)
        pad_w = jnp.zeros(pad_size, dtype=ball_w.dtype)
        pad_h = jnp.zeros(pad_size, dtype=ball_h.dtype)
        pad_active = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)

        return LabeledObjectObservation(
            x=jnp.concatenate([ball_x, pad_x]),
            y=jnp.concatenate([ball_y, pad_y]),
            width=jnp.concatenate([ball_w, pad_w]),
            height=jnp.concatenate([ball_h, pad_h]),
            active=jnp.concatenate([ball_active, pad_active]),
            semantic_label=jnp.concatenate([ball_label, pad_labels]),
            state=jnp.zeros(1 + pad_size, dtype=jnp.int32),
            can_go_offscreen=jnp.zeros(1 + pad_size, dtype=jnp.int32),
            was_killed=jnp.zeros(1 + pad_size, dtype=jnp.int32),
        )
