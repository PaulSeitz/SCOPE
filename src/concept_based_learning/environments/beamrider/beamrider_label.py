import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
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


# Beamrider: full 18-action set. Player moves on a fixed horizontal lane,
# shooting upward (FIRE). Target-aligned ego obs works with 8-directional movement.
beamrider_action_mapper, beamrider_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=     [3,  8, 5,  9, 4, 7, 2, 6],   # R DR D DL L UL U UR
    move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
)

# Object counts (from BeamriderConstants defaults):
#   white_ufo=3, enemy_shots=9, chasing_meteoroids=6, falling_rocks=3,
#   lane_blockers=1, coins=3, kamikaze=1, rejuvenator=1, bouncer=1, mothership=1
# Total = 29 objects
_BEAMRIDER_MAX_OBJECTS = 31  # 29 game + 1 NOOP + 1 FIRE

# Observation slot layout (game_max_objects=29, +2 meta slots appended by wrapper):
#   DESTROY: [0-2] white_ufo(3), [3] kamikaze, [4] mothership
#   AVOID:   [5-10] chasing_meteoroids(6), [11-19] enemy_shots(9),
#            [20-22] falling_rocks(3), [23] lane_blockers, [24] bouncer
#   COLLECT: [25-27] coins(3), [28] rejuvenator
#            (rejuvenator → AVOID while shot/dead debris; slot index unchanged)
#   META:    [29] FIRE, [30] NOOP  (MAX_OBJECTS=31)
_N_UFO = 3
_N_KAMIKAZE = 1
_N_MOTHERSHIP = 1
_N_DESTROY = _N_UFO + _N_KAMIKAZE + _N_MOTHERSHIP  # 5
_N_METEORS = 6
_N_COINS = 3
_N_GAME_OBJECTS = 29


def _scalar_to_1d(x):
    """Ensure scalar field becomes shape (1,)."""
    return jnp.expand_dims(x, axis=0) if x.ndim == 0 else x


def rejuvenator_semantic_label(dead) -> jnp.ndarray:
    """Intact rejuvenator is COLLECT; shot debris is AVOID (touching kills)."""
    dead_b = jnp.asarray(dead).astype(jnp.bool_)
    if dead_b.ndim == 0:
        dead_b = dead_b[None]
    return jnp.where(
        dead_b,
        jnp.full(dead_b.shape, LABEL_AVOID, dtype=jnp.int32),
        jnp.full(dead_b.shape, LABEL_COLLECT, dtype=jnp.int32),
    )


def _explosion_rising_edge(prev_frame, next_frame):
    """Kill fired when explosion_frame goes 0 → >0 (hit registered this step)."""
    prev = prev_frame if prev_frame.ndim > 0 else jnp.expand_dims(prev_frame, 0)
    nxt = next_frame if next_frame.ndim > 0 else jnp.expand_dims(next_frame, 0)
    return jnp.logical_and(prev == 0, nxt > 0).astype(jnp.int32)


class BeamriderSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        super().__init__(env, max_objects=_BEAMRIDER_MAX_OBJECTS, meta_reserved_slots=2)

    def _substep_env_state_events(self, prev_env_state, next_env_state):
        """
        Detect DESTROY kills from explosion_frame rising edges in level state.

        ufo/kamikaze: explosion_frame 0→>0.
        mothership: mothership_stage >0→0 (reset to 0 on kill via dying sequence).
        Chasing meteoroids are AVOID (dodge-only); their explosions are not
        stamped as DESTROY kills.
        Returns a 29-element mask matching the observation slot layout.
        """
        pl = prev_env_state.level
        nl = next_env_state.level

        ufo_killed = _explosion_rising_edge(pl.ufo_explosion_frame, nl.ufo_explosion_frame)
        kamikaze_killed = _explosion_rising_edge(
            pl.kamikaze_explosion_frame, nl.kamikaze_explosion_frame
        )
        mothership_killed = jnp.logical_and(
            pl.mothership_stage > 0, nl.mothership_stage == 0
        ).astype(jnp.int32)[None]

        destroy_mask = jnp.concatenate(
            [ufo_killed, kamikaze_killed, mothership_killed]
        )  # (5,)
        avoid_collect_zeros = jnp.zeros(
            _N_GAME_OBJECTS - destroy_mask.shape[0], dtype=jnp.int32
        )
        return jnp.concatenate([destroy_mask, avoid_collect_zeros])  # (29,)

    def _stamp_events_on_observation(self, semantic_obs, kill_mask):
        if kill_mask is None:
            return semantic_obs
        was_killed_with_noop = self._with_meta_event_padding(kill_mask)
        objects = semantic_obs.objects
        new_wk = objects.was_killed.at[-1].set(was_killed_with_noop)
        return semantic_obs.replace(objects=objects.replace(was_killed=new_wk))

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        # --- DESTROY (shootable) ---
        ufo = obs.white_ufo  # n=3
        kamikaze = obs.kamikaze  # n=1
        mothership = obs.mothership  # n=1

        # --- AVOID (dodge; meteoroids are not laser targets) ---
        meteors = obs.chasing_meteoroids  # n=6
        enemy_shots = obs.enemy_shots  # n=9
        falling_rocks = obs.falling_rocks  # n=3
        lane_blockers = obs.lane_blockers  # n=1
        bouncer = obs.bouncer  # n=1

        # --- COLLECT ---
        coins = obs.coins  # n=3
        rejuvenator = obs.rejuvenator  # n=1

        # Flatten scalar-or-vector fields to 1D for single-object entries
        def _expand(obj):
            return (
                _scalar_to_1d(obj.x),
                _scalar_to_1d(obj.y),
                _scalar_to_1d(obj.width),
                _scalar_to_1d(obj.height),
                _scalar_to_1d(obj.active),
            )

        kam_x, kam_y, kam_w, kam_h, kam_act = _expand(kamikaze)
        ms_x, ms_y, ms_w, ms_h, ms_act = _expand(mothership)
        bou_x, bou_y, bou_w, bou_h, bou_act = _expand(bouncer)
        rej_x, rej_y, rej_w, rej_h, rej_act = _expand(rejuvenator)

        destroy_x = jnp.concatenate([ufo.x, kam_x, ms_x])
        destroy_y = jnp.concatenate([ufo.y, kam_y, ms_y])
        destroy_w = jnp.concatenate([ufo.width, kam_w, ms_w])
        destroy_h = jnp.concatenate([ufo.height, kam_h, ms_h])
        destroy_act = jnp.concatenate([ufo.active, kam_act, ms_act])
        n_destroy = destroy_x.shape[0]  # 3+1+1 = 5

        avoid_x = jnp.concatenate(
            [meteors.x, enemy_shots.x, falling_rocks.x, lane_blockers.x, bou_x]
        )
        avoid_y = jnp.concatenate(
            [meteors.y, enemy_shots.y, falling_rocks.y, lane_blockers.y, bou_y]
        )
        avoid_w = jnp.concatenate(
            [meteors.width, enemy_shots.width, falling_rocks.width, lane_blockers.width, bou_w]
        )
        avoid_h = jnp.concatenate(
            [meteors.height, enemy_shots.height, falling_rocks.height, lane_blockers.height, bou_h]
        )
        avoid_act = jnp.concatenate(
            [
                meteors.active,
                enemy_shots.active,
                falling_rocks.active,
                lane_blockers.active,
                bou_act,
            ]
        )
        n_avoid = avoid_x.shape[0]  # 6+9+3+1+1 = 20

        collect_x = jnp.concatenate([coins.x, rej_x])
        collect_y = jnp.concatenate([coins.y, rej_y])
        collect_w = jnp.concatenate([coins.width, rej_w])
        collect_h = jnp.concatenate([coins.height, rej_h])
        collect_act = jnp.concatenate([coins.active, rej_act])

        # Shot rejuvenator is hostile debris (touch = death) but keeps its slot.
        rej_dead = _scalar_to_1d(
            getattr(rejuvenator, "visual_id", getattr(obs, "rejuvenator_dead", 0))
        )
        collect_labels = jnp.concatenate(
            [
                jnp.full(int(coins.x.shape[0]), LABEL_COLLECT, dtype=jnp.int32),
                rejuvenator_semantic_label(rej_dead).reshape((1,)),
            ]
        )

        all_x = jnp.concatenate([destroy_x, avoid_x, collect_x])
        all_y = jnp.concatenate([destroy_y, avoid_y, collect_y])
        all_w = jnp.concatenate([destroy_w, avoid_w, collect_w])
        all_h = jnp.concatenate([destroy_h, avoid_h, collect_h])
        all_active = jnp.concatenate([destroy_act, avoid_act, collect_act])
        all_labels = jnp.concatenate([
            jnp.full(n_destroy, LABEL_DESTROY, dtype=jnp.int32),
            jnp.full(n_avoid, LABEL_AVOID, dtype=jnp.int32),
            collect_labels,
        ])
        assert all_labels.shape[0] == all_x.shape[0]

        n_objects = all_x.shape[0]  # 29
        pad_size = self.game_max_objects - n_objects
        pad_x = jnp.zeros(pad_size, dtype=all_x.dtype)
        pad_y = jnp.zeros(pad_size, dtype=all_y.dtype)
        pad_w = jnp.zeros(pad_size, dtype=all_w.dtype)
        pad_h = jnp.zeros(pad_size, dtype=all_h.dtype)
        pad_active = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)

        # All beamrider objects can scroll or move off screen edges
        all_offscreen = jnp.ones(n_objects, dtype=jnp.int32)
        pad_offscreen = jnp.zeros(pad_size, dtype=jnp.int32)

        return LabeledObjectObservation(
            x=jnp.concatenate([all_x, pad_x]),
            y=jnp.concatenate([all_y, pad_y]),
            width=jnp.concatenate([all_w, pad_w]),
            height=jnp.concatenate([all_h, pad_h]),
            active=jnp.concatenate([all_active, pad_active]),
            semantic_label=jnp.concatenate([all_labels, pad_labels]),
            state=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
            can_go_offscreen=jnp.concatenate([all_offscreen, pad_offscreen]),
            was_killed=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
        )