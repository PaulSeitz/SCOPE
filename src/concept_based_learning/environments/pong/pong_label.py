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


# Pong: player at x=140 (right side). RIGHT action moves paddle up (negative y),
# LEFT action moves paddle down (positive y) — confusingly inverted in jax_pong.py.
# Player moves only vertically, so we map the vertical ego directions to RIGHT/LEFT.
# Horizontal ego directions are NOOP (player cannot move horizontally).
# R DR D DL L  UL U  UR
# RIGHT is "up" (negative y), LEFT is "down" (positive y)
pong_action_mapper, pong_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=     [0, 3, 3, 3, 0, 2, 2, 2],  # R DR D DL L UL U UR
    move_fire_table=[5, 3, 3, 3, 5, 2, 2, 2],
)


class PongSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # 1 ball = 1 object + padding to 2 (slot 0 is meta-NOOP)
        super().__init__(env, max_objects=2)

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        ball = obs.ball

        ball_x = jnp.expand_dims(ball.x, axis=0) if ball.x.ndim == 0 else ball.x
        ball_y = jnp.expand_dims(ball.y, axis=0) if ball.y.ndim == 0 else ball.y
        ball_w = jnp.expand_dims(ball.width, axis=0) if ball.width.ndim == 0 else ball.width
        ball_h = jnp.expand_dims(ball.height, axis=0) if ball.height.ndim == 0 else ball.height

        # Ball active: it resets to center after a point, it never truly disappears.
        # Use AABB overlap (rising-edge) for COLLECT reward — ball never gets destroyed.
        ball_active = jnp.ones(1, dtype=jnp.int32)
        ball_label = jnp.array([LABEL_COLLECT], dtype=jnp.int32)

        pad_size = self.game_max_objects - 1
        pad_x = jnp.zeros(pad_size, dtype=ball_x.dtype)
        pad_y = jnp.zeros(pad_size, dtype=ball_y.dtype)
        pad_w = jnp.zeros(pad_size, dtype=ball_w.dtype)
        pad_h = jnp.zeros(pad_size, dtype=ball_h.dtype)
        pad_active = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)
        pad_offscreen = jnp.zeros(pad_size, dtype=jnp.int32)

        return LabeledObjectObservation(
            x=jnp.concatenate([ball_x, pad_x]),
            y=jnp.concatenate([ball_y, pad_y]),
            width=jnp.concatenate([ball_w, pad_w]),
            height=jnp.concatenate([ball_h, pad_h]),
            active=jnp.concatenate([ball_active, pad_active]),
            semantic_label=jnp.concatenate([ball_label, pad_labels]),
            state=jnp.zeros(1 + pad_size, dtype=jnp.int32),
            can_go_offscreen=jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), pad_offscreen]),
            was_killed=jnp.zeros(1 + pad_size, dtype=jnp.int32),
        )
