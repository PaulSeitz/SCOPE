import jax.numpy as jnp
from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.observations.action_space import make_ego_action_mapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
    LABEL_NOOP,
)

from concept_based_learning.environments.skill_rewards import extract_skill_rewards

# SpaceInvaders screen boundaries — enemies and bullets scroll off the side edges.
def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost):
    del next_raw_obs
    return extract_skill_rewards(
        prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost
    )


# SpaceInvaders: purely horizontal movement, 6 actions (0=NOOP,1=FIRE,2=RIGHT,3=LEFT,4=RIGHTFIRE,5=LEFTFIRE).
# Direction table maps the 8 world directions to horizontal actions; UP/DOWN collapse to NOOP/FIRE.
spaceinvaders_action_mapper = make_ego_action_mapper(
    noop_action=0,
    fire_action=1,
    move_table=     [2, 2, 0, 3, 3, 3, 0, 2],   # R DR D DL L UL U UR
    move_fire_table=[4, 4, 1, 5, 5, 5, 1, 4],
)


class SpaceInvadersSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # 36 Enemies + 3 Enemy Bullets + 1 UFO = 40 active objects
        super().__init__(env, max_objects=41)

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        """
        Takes a single Jaxatari SpaceInvaders observation, strips irrelevant entities,
        assigns universal labels, and flattens it into our universal object list.
        """
        enemies = obs.enemies                 # n=36
        enemy_bullets = obs.enemy_bullets     # n=3
        ufo = obs.ufo                         # n=1

        # Note: We skip `player_bullet` because we don't want the agent to target its own bullet

        # Create labels
        enemy_labels = jnp.full(36, LABEL_DESTROY, dtype=jnp.int32)
        enemy_bullet_labels = jnp.full(3, LABEL_AVOID, dtype=jnp.int32)
        ufo_label = jnp.array([LABEL_DESTROY], dtype=jnp.int32)

        # Concatenate features
        def extract_ufo_field(field):
            return jnp.expand_dims(field, axis=0) if field.ndim == 0 else field

        ufo_x = extract_ufo_field(ufo.x)
        ufo_y = extract_ufo_field(ufo.y)
        ufo_w = extract_ufo_field(ufo.width)
        ufo_h = extract_ufo_field(ufo.height)
        ufo_active = extract_ufo_field(ufo.active)

        all_x = jnp.concatenate([enemies.x, enemy_bullets.x, ufo_x])
        all_y = jnp.concatenate([enemies.y, enemy_bullets.y, ufo_y])
        all_w = jnp.concatenate([enemies.width, enemy_bullets.width, ufo_w])
        all_h = jnp.concatenate([enemies.height, enemy_bullets.height, ufo_h])
        all_active = jnp.concatenate([enemies.active, enemy_bullets.active, ufo_active])
        all_labels = jnp.concatenate([enemy_labels, enemy_bullet_labels, ufo_label])
        all_offscreen = jnp.ones(all_x.shape[0], dtype=jnp.int32)  # all objects scroll off edges

        return LabeledObjectObservation(
            x=all_x,
            y=all_y,
            width=all_w,
            height=all_h,
            active=all_active,
            semantic_label=all_labels,
            can_go_offscreen=all_offscreen,
            was_killed=jnp.zeros_like(all_offscreen),
        )
