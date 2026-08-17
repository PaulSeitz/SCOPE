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


# Tennis: full 18-action set. Player can move in all directions and swing with FIRE.
# Target-aligned ego obs works naturally with 8-directional movement.
tennis_action_mapper, tennis_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=     [3,  8, 5,  9, 4, 7, 2, 6],   # R DR D DL L UL U UR
    move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
)


def _apply_visual_ball_y(obs, ball_z):
    """
    Replace ball.y (court/shadow position) with visual y (ball_y - ball_z).

    The TennisObservation stores ball_y as the court projection (shadow).
    The actual rendered ball is at ball_y - ball_z. We use visual y for AABB
    so that COLLECT fires when the player is at the ball, not the shadow.

    obs.raw_observation.ball.y has shape [stack_size]; ball_z is a scalar from
    the current env_state. All stack frames are corrected by the same ball_z
    (only the most-recent frame's height is available).
    """
    screen_h = jnp.array(209, dtype=jnp.int32)
    visual_y = jnp.clip(
        (obs.raw_observation.ball.y - ball_z.astype(jnp.int32)),
        0,
        screen_h,
    )
    new_ball = obs.raw_observation.ball.replace(y=visual_y)
    new_raw = obs.raw_observation.replace(ball=new_ball)
    return obs.replace(raw_observation=new_raw)


class TennisSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # 1 ball = 1 object + padding to 2
        super().__init__(env, max_objects=2)

    def _patch_and_reprocess(self, obs, env_state):
        # Pixel wrappers nest AtariState under PixelAndObjectCentricState.atari_state
        # (no .env_state on that object). Unwrap to the JAXAtari Tennis env_state.
        game_state = self._unwrap_env_state(env_state.atari_state)
        ball_z = game_state.ball_state.ball_z
        obs = _apply_visual_ball_y(obs, ball_z)
        # Re-run object extraction on patched raw obs so objects.y uses visual y.
        semantic = self._process_stack(obs.raw_observation)
        return obs.replace(objects=semantic.objects)

    def step(self, state, action):
        obs, env_state, reward, done, info = super().step(state, action)
        obs = self._patch_and_reprocess(obs, env_state)
        return obs, env_state, reward, done, info

    def reset(self, key=None):
        obs, env_state = super().reset(key)
        obs = self._patch_and_reprocess(obs, env_state)
        return obs, env_state

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        ball = obs.ball

        ball_x = jnp.expand_dims(ball.x, axis=0) if ball.x.ndim == 0 else ball.x
        ball_y = jnp.expand_dims(ball.y, axis=0) if ball.y.ndim == 0 else ball.y
        ball_w = jnp.expand_dims(ball.width, axis=0) if ball.width.ndim == 0 else ball.width
        ball_h = jnp.expand_dims(ball.height, axis=0) if ball.height.ndim == 0 else ball.height

        ball_active = jnp.ones(1, dtype=jnp.int32)
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
