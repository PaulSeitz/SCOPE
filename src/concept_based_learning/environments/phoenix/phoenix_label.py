import jax
import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.observations.action_space import make_ego_action_mapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
    LABEL_AVOID,
    LABEL_NOOP,
)
from concept_based_learning.environments.skill_rewards import (
    NEAR_PLAYER_TOLERANCE,
    pack_skill_rewards,
    reward_transition_context,
    standard_avoid_outcomes,
    standard_destroy_outcomes,
)

# jax_phoenix observation encoding: base state 0..3, +4 while death animation plays.
DYING_STATE_THRESHOLD = 4
BOSS_WIDTH = 32
BOSS_HEIGHT = 16
# Playfield is 160x210; treat spawns/despawns at the border as non-rewards.
SCREEN_X_MIN = 2.0
SCREEN_X_MAX = 157.0
SCREEN_Y_MIN = 2.0
SCREEN_Y_MAX = 208.0


def _bbox_overlap_xywh(ax, ay, aw, ah, bx, by, bw, bh):
    a_left, a_right = ax, ax + aw
    a_top, a_bottom = ay, ay + ah
    b_left, b_right = bx, bx + bw
    b_top, b_bottom = by, by + bh
    horiz = jnp.logical_and(a_left <= b_right, a_right >= b_left)
    vert = jnp.logical_and(a_top <= b_bottom, a_bottom >= b_top)
    return jnp.logical_and(horiz, vert)


def _is_boss_target(target):
    return jnp.logical_and(target.width == BOSS_WIDTH, target.height == BOSS_HEIGHT)


def _entered_death_animation(prev_target, next_target):
    """Enemy entered jax_phoenix death animation (state crosses into >= 4)."""
    prev_state = prev_target.state
    next_state = next_target.state
    prev_dying = prev_state >= DYING_STATE_THRESHOLD
    next_dying = next_state >= DYING_STATE_THRESHOLD
    return jnp.logical_and(jnp.logical_not(prev_dying), next_dying)


def _player_projectile_hit(prev_raw_obs, target):
    """True if the player shot overlapped the target on the previous frame."""
    projectile = prev_raw_obs.player_projectile

    def last_scalar(field):
        val = getattr(projectile, field)
        return val[-1] if jnp.ndim(val) > 0 else val

    if jnp.ndim(projectile.active) > 0:
        active = projectile.active[-1] == 1
        px = projectile.x[-1]
        py = projectile.y[-1]
        pw = projectile.width[-1]
        ph = projectile.height[-1]
    else:
        active = projectile.active == 1
        px = projectile.x
        py = projectile.y
        pw = projectile.width
        ph = projectile.height

    return jnp.logical_and(
        active,
        _bbox_overlap_xywh(
            px,
            py,
            pw,
            ph,
            target.x,
            target.y,
            target.width,
            target.height,
        ),
    )


def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost):
    """
    Returns (rewards, skill_done). skill_done[k] is True only when the transition
    is terminal for skill k because the assigned target caused it (golden rule).
    """
    del next_raw_obs

    (
        _prev_player,
        prev_target,
        next_target,
        disappeared,
        hit_by_assigned,
        _near_player,
        at_edge,
    ) = reward_transition_context(
        prev_raw_obs,
        prev_target,
        next_target,
        near_tolerance=NEAR_PLAYER_TOLERANCE,
        x_min=SCREEN_X_MIN,
        x_max=SCREEN_X_MAX,
        y_min=SCREEN_Y_MIN,
        y_max=SCREEN_Y_MAX,
    )

    is_boss = _is_boss_target(prev_target)

    # Object-relative kill detection (hindsight relabeling): scored for the
    # selected object regardless of its label. Non-enemy objects never enter the
    # death-animation state and are rarely hit by the player shot, so these
    # predicates stay naturally enemy-specific without an explicit label gate.

    # Primary: reward on kill frame (death animation starts), not ~30 frames later.
    enemy_kill = jnp.logical_and(
        jnp.logical_not(is_boss),
        jnp.logical_and(
            _entered_death_animation(prev_target, next_target),
            jnp.logical_not(hit_by_assigned),
        ),
    )

    # Boss has no dying-state encoding; core hit deactivates immediately.
    boss_kill = jnp.logical_and(
        is_boss,
        jnp.logical_and(disappeared, jnp.logical_not(hit_by_assigned)),
    )

    # Fallback: frame-skip may skip the exact dying transition but keep projectile overlap.
    projectile_kill = jnp.logical_and(
        jnp.logical_not(is_boss),
        jnp.logical_and(
            disappeared,
            jnp.logical_and(
                _player_projectile_hit(prev_raw_obs, prev_target),
                jnp.logical_and(jnp.logical_not(at_edge), jnp.logical_not(hit_by_assigned)),
            ),
        ),
    )

    phoenix_destroy_success = jnp.logical_or(
        enemy_kill, jnp.logical_or(boss_kill, projectile_kill)
    )
    destroy_success, destroy_failure = standard_destroy_outcomes(
        prev_target,
        disappeared=disappeared,
        hit_by_assigned=hit_by_assigned,
        at_screen_edge=at_edge,
        life_lost=life_lost,
        destroy_success=phoenix_destroy_success,
    )
    avoid_success, avoid_failure = standard_avoid_outcomes(
        prev_target,
        disappeared=disappeared,
        hit_by_assigned=hit_by_assigned,
        life_lost=life_lost,
    )
    return pack_skill_rewards(
        destroy_success=destroy_success,
        destroy_failure=destroy_failure,
        collect_success=jnp.array(False),
        avoid_success=avoid_success,
        avoid_failure=avoid_failure,
    )


# Phoenix: horizontal movement + shield (DOWN); ship fires upward so all *_FIRE collapses to FIRE.
# Direction table (R DR D DL L UL U UR) → (RIGHT, RIGHT, DOWN/shield, LEFT, LEFT, LEFT, NOOP, RIGHT)
phoenix_action_mapper = make_ego_action_mapper(
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

        # Only enemies use the special state encoding; others get a neutral default (0).
        enemy_state = enemies.state
        zeros_enemy_bullets_state = jnp.zeros_like(enemy_bullets.x, dtype=jnp.int32)
        zeros_boss_state = jnp.zeros_like(boss_x, dtype=jnp.int32)

        all_x = jnp.concatenate([enemies.x, enemy_bullets.x, boss_x])
        all_y = jnp.concatenate([enemies.y, enemy_bullets.y, boss_y])
        all_w = jnp.concatenate([enemies.width, enemy_bullets.width, boss_w])
        all_h = jnp.concatenate([enemies.height, enemy_bullets.height, boss_h])
        all_active = jnp.concatenate([enemies.active, enemy_bullets.active, boss_active])
        all_labels = jnp.concatenate([enemy_labels, enemy_bullet_labels, boss_label])
        all_state = jnp.concatenate(
            [enemy_state, zeros_enemy_bullets_state, zeros_boss_state]
        )

        pad_size = self.game_max_objects - all_x.shape[0]
        pad_x = jnp.zeros(pad_size, dtype=all_x.dtype)
        pad_y = jnp.zeros(pad_size, dtype=all_y.dtype)
        pad_w = jnp.zeros(pad_size, dtype=all_w.dtype)
        pad_h = jnp.zeros(pad_size, dtype=all_h.dtype)
        pad_active = jnp.zeros(pad_size, dtype=all_active.dtype)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)
        pad_state = jnp.zeros(pad_size, dtype=all_state.dtype)

        return LabeledObjectObservation(
            x=jnp.concatenate([all_x, pad_x]),
            y=jnp.concatenate([all_y, pad_y]),
            width=jnp.concatenate([all_w, pad_w]),
            height=jnp.concatenate([all_h, pad_h]),
            active=jnp.concatenate([all_active, pad_active]),
            semantic_label=jnp.concatenate([all_labels, pad_labels]),
            state=jnp.concatenate([all_state, pad_state]),
        )
