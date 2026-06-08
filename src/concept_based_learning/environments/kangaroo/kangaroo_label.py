import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_DESTROY,
    LABEL_COLLECT,
    LABEL_AVOID,
    LABEL_NOOP,
)
from concept_based_learning.environments.skill_rewards import extract_skill_rewards
from concept_based_learning.observations.action_space import make_ego_action_mapper


def extract_all_skills_rewards(prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost):
    return extract_skill_rewards(
        prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost
    )


# Kangaroo: full 18-action set. Player moves in all 8 directions + jump/punch (FIRE).
kangaroo_action_mapper = make_ego_action_mapper(
    noop_action=0,
    fire_action=1,
    move_table=     [3,  8, 5,  9, 4, 7, 2, 6],   # R DR D DL L UL U UR
    move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
)

# Each ladder produces two virtual 1px COLLECT waypoints:
#   bottom: (ladder_x + ladder_w//2,  ladder_y + ladder_h)
#   top:    (ladder_x + ladder_w//2,  ladder_y)
# Observation pads ladders to 20 slots; inactive slots (active=0) are ignored by meta.
#
# Total objects: 20 bottom + 20 top + 3 fruits + 1 bell + 1 child
#              + 1 falling_coconut + 4 monkeys + 4 thrown_coconuts = 54
_KANGAROO_MAX_OBJECTS = 55  # 54 + 1 padding


def _scalar_to_1d(x):
    return jnp.expand_dims(x, axis=0) if x.ndim == 0 else x


class KangarooSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        super().__init__(env, max_objects=_KANGAROO_MAX_OBJECTS)

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        ladders = obs.ladders               # n=20 (padded)
        fruits = obs.fruits                 # n=3
        bell = obs.bell                     # scalar
        child = obs.child                   # scalar
        falling_coconut = obs.falling_coconut  # scalar
        monkeys = obs.monkeys               # n=4
        thrown_coconuts = obs.thrown_coconuts  # n=4

        # --- Ladder waypoints ---
        lad_half_w = ladders.width // 2
        lad_cx = ladders.x + lad_half_w  # center x of each ladder

        # Bottom waypoints: y = ladder_y + ladder_h
        bot_x = lad_cx
        bot_y = ladders.y + ladders.height
        bot_active = ladders.active

        # Top waypoints: y = ladder_y
        top_x = lad_cx
        top_y = ladders.y
        top_active = ladders.active

        waypoint_w = jnp.ones_like(bot_x)
        waypoint_h = jnp.ones_like(bot_y)
        n_ladders = ladders.x.shape[0]  # 20

        # --- Flatten scalar fields ---
        bell_x, bell_y = _scalar_to_1d(bell.x), _scalar_to_1d(bell.y)
        bell_w, bell_h = _scalar_to_1d(bell.width), _scalar_to_1d(bell.height)
        bell_act = _scalar_to_1d(bell.active)

        child_x, child_y = _scalar_to_1d(child.x), _scalar_to_1d(child.y)
        child_w, child_h = _scalar_to_1d(child.width), _scalar_to_1d(child.height)
        child_act = _scalar_to_1d(child.active)

        fc_x, fc_y = _scalar_to_1d(falling_coconut.x), _scalar_to_1d(falling_coconut.y)
        fc_w, fc_h = _scalar_to_1d(falling_coconut.width), _scalar_to_1d(falling_coconut.height)
        fc_act = _scalar_to_1d(falling_coconut.active)

        # --- Concatenate by label group ---
        collect_x = jnp.concatenate([bot_x, top_x, fruits.x, bell_x, child_x])
        collect_y = jnp.concatenate([bot_y, top_y, fruits.y, bell_y, child_y])
        collect_w = jnp.concatenate([waypoint_w, waypoint_w, fruits.width, bell_w, child_w])
        collect_h = jnp.concatenate([waypoint_h, waypoint_h, fruits.height, bell_h, child_h])
        collect_act = jnp.concatenate([bot_active, top_active, fruits.active, bell_act, child_act])
        n_collect = collect_x.shape[0]  # 20+20+3+1+1 = 45

        destroy_x = jnp.concatenate([fc_x, monkeys.x])
        destroy_y = jnp.concatenate([fc_y, monkeys.y])
        destroy_w = jnp.concatenate([fc_w, monkeys.width])
        destroy_h = jnp.concatenate([fc_h, monkeys.height])
        destroy_act = jnp.concatenate([fc_act, monkeys.active])
        n_destroy = destroy_x.shape[0]  # 1+4 = 5

        avoid_x = thrown_coconuts.x
        avoid_y = thrown_coconuts.y
        avoid_w = thrown_coconuts.width
        avoid_h = thrown_coconuts.height
        avoid_act = thrown_coconuts.active
        n_avoid = avoid_x.shape[0]  # 4

        all_x = jnp.concatenate([collect_x, destroy_x, avoid_x])
        all_y = jnp.concatenate([collect_y, destroy_y, avoid_y])
        all_w = jnp.concatenate([collect_w, destroy_w, avoid_w])
        all_h = jnp.concatenate([collect_h, destroy_h, avoid_h])
        all_active = jnp.concatenate([collect_act, destroy_act, avoid_act])
        all_labels = jnp.concatenate([
            jnp.full(n_collect, LABEL_COLLECT, dtype=jnp.int32),
            jnp.full(n_destroy, LABEL_DESTROY, dtype=jnp.int32),
            jnp.full(n_avoid,   LABEL_AVOID,   dtype=jnp.int32),
        ])

        n_objects = all_x.shape[0]  # 54
        pad_size = self.game_max_objects - n_objects
        pad_x = jnp.zeros(pad_size, dtype=all_x.dtype)
        pad_y = jnp.zeros(pad_size, dtype=all_y.dtype)
        pad_w = jnp.zeros(pad_size, dtype=all_w.dtype)
        pad_h = jnp.zeros(pad_size, dtype=all_h.dtype)
        pad_active = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)

        # Kangaroo objects don't scroll off screen edges (ladders, monkeys, coconuts
        # are contained within the playfield and disappear in-place).
        all_offscreen = jnp.zeros(n_objects, dtype=jnp.int32)
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
