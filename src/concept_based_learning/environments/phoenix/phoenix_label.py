import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.observations.action_space import make_target_aligned_action_mappers
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
    LABEL_AVOID,
    LABEL_NOOP,
)
from concept_based_learning.environments.skill_rewards import (
    avoid_survival_reward_from_config,
    extract_skill_rewards,
    penalize_collect_death_from_config,
)

# jax_phoenix observation encoding: base state 0..3, +4 while death animation plays.
# We set active=0 the moment state >= DYING_STATE_THRESHOLD so meta-policy
# deprioritises dying enemies immediately.  DESTROY rewards use was_killed from
# phoenix_dying/bat_dying rising edges in _substep_env_state_events.
DYING_STATE_THRESHOLD = 4
# Playfield is 160px wide; ignore kills whose previous x was at the scroll edge.
_SCROLL_X_MIN = 5
_SCROLL_X_MAX = 155


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


# Phoenix: horizontal movement + shield (DOWN); ship fires upward so all *_FIRE collapses to FIRE.
# Direction table (R DR D DL L UL U UR) → (RIGHT, RIGHT, DOWN/shield, LEFT, LEFT, LEFT, NOOP, RIGHT)
phoenix_action_mapper, phoenix_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=     [2, 2, 4, 3, 3, 3, 0, 2],
    move_fire_table=[1, 1, 1, 1, 1, 1, 1, 1],
)


# Targetable entities only (boss blocks are structural LABEL_NOOP and are omitted).
PHOENIX_MAX_OBJECTS = 21


class PhoenixSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        # 8 enemies + 8 bullets + 1 boss (+ padding to PHOENIX_MAX_OBJECTS)
        super().__init__(env, max_objects=PHOENIX_MAX_OBJECTS)

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """
        Detect enemy kills this substep.

        Kill signal: an enemy newly enters dying state (phoenix_dying or bat_dying
        goes False → True). Boss kill: boss.active goes True → False.
        Returns a 17-element mask: [enemies(8) | bullets(8) | boss(1)].
        """
        prev_dying = prev_env_state.phoenix_dying | prev_env_state.bat_dying
        next_dying = next_env_state.phoenix_dying | next_env_state.bat_dying
        enemy_killed = jnp.logical_and(~prev_dying, next_dying)
        prev_x = prev_env_state.enemies_x
        on_screen = jnp.logical_and(prev_x > _SCROLL_X_MIN, prev_x < _SCROLL_X_MAX)
        enemy_killed = jnp.logical_and(enemy_killed, on_screen).astype(jnp.int32)

        bullet_killed = jnp.zeros(8, dtype=jnp.int32)

        boss_killed = jnp.logical_and(
            prev_env_state.boss.active, ~next_env_state.boss.active
        ).astype(jnp.int32)
        boss_killed = jnp.expand_dims(boss_killed, axis=0)

        return jnp.concatenate([enemy_killed, bullet_killed, boss_killed])  # (17,)

    def _stamp_events_on_observation(self, semantic_obs, kill_mask):
        if kill_mask is None:
            return semantic_obs
        # Game layout: [enemies(8) | bullets(8) | boss(1)] = 17 slots, padded to
        # game_max_objects=20, then +1 meta NOOP → was_killed shape [stack, 21].
        pad = jnp.zeros(self.game_max_objects - kill_mask.shape[0], dtype=jnp.int32)
        was_killed_with_noop = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), kill_mask.astype(jnp.int32), pad]
        )
        objects = semantic_obs.objects
        new_wk = objects.was_killed.at[-1].set(was_killed_with_noop)
        return semantic_obs.replace(objects=objects.replace(was_killed=new_wk))

    def step(self, state, action):
        """Override step to trick the network's orientation distribution."""
        obs, env_state, reward, done, info = super().step(state, action)

        # Stabilize orientation for target-aligned ego action mapping (ship faces upward).
        new_player = obs.raw_observation.player.replace(
            orientation=jnp.full_like(obs.raw_observation.player.orientation, 90.0)
        )
        new_raw_obs = obs.raw_observation.replace(player=new_player)
        obs = obs.replace(raw_observation=new_raw_obs)

        return obs, env_state, reward, done, info

    def reset(self, key=None):
        """Override reset to trick the network's orientation distribution."""
        obs, env_state = super().reset(key)

        new_player = obs.raw_observation.player.replace(
            orientation=jnp.full_like(obs.raw_observation.player.orientation, 90.0)
        )
        new_raw_obs = obs.raw_observation.replace(player=new_player)
        obs = obs.replace(raw_observation=new_raw_obs)

        return obs, env_state

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        enemies = obs.enemies
        enemy_bullets = obs.enemy_projectiles
        boss = obs.boss

        num_enemies = enemies.x.shape[-1]
        num_enemy_bullets = enemy_bullets.x.shape[-1]

        enemy_labels = jnp.full(num_enemies, LABEL_DESTROY, dtype=jnp.int32)
        enemy_bullet_labels = jnp.full(num_enemy_bullets, LABEL_AVOID, dtype=jnp.int32)
        boss_label = jnp.array([LABEL_DESTROY], dtype=jnp.int32)

        def extract_single_field(field):
            return jnp.expand_dims(field, axis=0) if field.ndim == 0 else field

        boss_x = extract_single_field(boss.x)
        boss_y = extract_single_field(boss.y)
        boss_w = extract_single_field(boss.width)
        boss_h = extract_single_field(boss.height)
        boss_active = extract_single_field(boss.active)

        # Mark enemy active=0 the moment the death animation starts (state >= 4)
        # so the meta-policy stops selecting dying enemies before sprite removal.
        enemy_alive = (enemies.state < DYING_STATE_THRESHOLD).astype(enemies.active.dtype)
        enemy_active = jnp.logical_and(enemies.active, enemy_alive).astype(enemies.active.dtype)

        # Only enemies use the special state encoding; others get a neutral default (0).
        enemy_state = enemies.state
        zeros_enemy_bullets_state = jnp.zeros_like(enemy_bullets.x, dtype=jnp.int32)
        zeros_boss_state = jnp.zeros_like(boss_x, dtype=jnp.int32)

        all_x = jnp.concatenate([enemies.x, enemy_bullets.x, boss_x])
        all_y = jnp.concatenate([enemies.y, enemy_bullets.y, boss_y])
        all_w = jnp.concatenate([enemies.width, enemy_bullets.width, boss_w])
        all_h = jnp.concatenate([enemies.height, enemy_bullets.height, boss_h])
        all_active = jnp.concatenate([enemy_active, enemy_bullets.active, boss_active])
        all_labels = jnp.concatenate([enemy_labels, enemy_bullet_labels, boss_label])
        all_state = jnp.concatenate(
            [enemy_state, zeros_enemy_bullets_state, zeros_boss_state]
        )
        # All Phoenix objects fly across the playfield and can reach the screen edges.
        all_offscreen = jnp.ones(all_x.shape[0], dtype=jnp.int32)

        pad_size = self.game_max_objects - all_x.shape[0]
        pad_x = jnp.zeros(pad_size, dtype=all_x.dtype)
        pad_y = jnp.zeros(pad_size, dtype=all_y.dtype)
        pad_w = jnp.zeros(pad_size, dtype=all_w.dtype)
        pad_h = jnp.zeros(pad_size, dtype=all_h.dtype)
        pad_active = jnp.zeros(pad_size, dtype=all_active.dtype)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)
        pad_state = jnp.zeros(pad_size, dtype=all_state.dtype)
        pad_offscreen = jnp.zeros(pad_size, dtype=jnp.int32)

        return LabeledObjectObservation(
            x=jnp.concatenate([all_x, pad_x]),
            y=jnp.concatenate([all_y, pad_y]),
            width=jnp.concatenate([all_w, pad_w]),
            height=jnp.concatenate([all_h, pad_h]),
            active=jnp.concatenate([all_active, pad_active]),
            semantic_label=jnp.concatenate([all_labels, pad_labels]),
            state=jnp.concatenate([all_state, pad_state]),
            can_go_offscreen=jnp.concatenate([all_offscreen, pad_offscreen]),
            was_killed=jnp.zeros(all_x.shape[0] + pad_size, dtype=jnp.int32),
        )
