"""Montezuma's Revenge semantic labeling.

Observation entities (jax_montezumarevenge):
  enemies, items, doors, ropes, conveyors, platforms.

Ladders exist only in env state (not in the object observation).  This wrapper
injects virtual COLLECT waypoints from ``env_state.ladders_*`` (Kangaroo
pattern) and patches player leaves that jaxatari emits as length-1 vectors.

jaxatari renders room sprites at ``state_y + 47`` (HUD band). This wrapper
shifts player and object Y by the same offset so overlays and ego features
align with the rendered frame.

Lasers exist in env state but are not exposed in the object observation, so
they are omitted here.

Layout (game_max_objects=31, +1 meta NOOP → 32):
  [0-2]   enemies           (AVOID — skull)
  [3-6]   ladder bottoms    (COLLECT waypoints; 0=center, 1=right, 2=left)
  [7-10]  ladder tops       (COLLECT waypoints; spawn-adjacent tops start inactive)
  [11-13] items             (COLLECT)
  [14-15] doors             (COLLECT — showcase merges NAVIGATE into COLLECT)
  [16-17] ropes             (COLLECT grab WPs near rope top — optional / return)
  [18]    conveyors         (BLOCKER)
  [19-30] platforms         (BLOCKER; room-4 also hosts shaft bead + edge
                            COLLECT; gap slots parked off-screen)

Room-4 path scaffolding (obs coords; see ``montezuma_policy.WP*``):
  platform / ladder slots stamped from numbered waypoint centres so the
  policy file is the single place to retune trigger positions.

With meta-NOOP prepended, planner slots are +1 (enemies start at 1).
"""

from __future__ import annotations

import jax.numpy as jnp

from concept_based_learning.environments.labeling_wrapper import JaxatariSemanticBaseWrapper
from concept_based_learning.environments.datatypes import (
    LabeledObjectObservation,
    LABEL_AVOID,
    LABEL_BLOCKER,
    LABEL_COLLECT,
    LABEL_NOOP,
)
from concept_based_learning.environments.skill_rewards import (
    avoid_survival_reward_from_config,
    extract_skill_rewards,
    penalize_collect_death_from_config,
)
from concept_based_learning.observations.action_space import make_target_aligned_action_mappers


def extract_all_skills_rewards(
    prev_raw_obs, next_raw_obs, prev_target, next_target, life_lost, info=None, config=None
):
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


# Full 18-action set (FIRE = jump). Same layout as Kangaroo / Beamrider.
montezuma_action_mapper, montezuma_inverse_action_mapper = make_target_aligned_action_mappers(
    noop_action=0,
    fire_action=1,
    move_table=[3, 8, 5, 9, 4, 7, 2, 6],  # R DR D DL L UL U UR
    move_fire_table=[11, 16, 13, 17, 12, 15, 10, 14],
)

# Stamped on ``raw_observation.player.state[-1]`` while ``death_timer > 0``.
# The Sequence policy freezes plan_idx at 0 during this window (Kangaroo pattern).
PLAYER_STATE_DYING = 1

_N_ENEMIES = 3
_N_LADDERS = 4  # MAX_LADDERS_PER_ROOM
_N_WAYPOINTS = _N_LADDERS * 2  # bottom then top
_N_ITEMS = 3
_N_DOORS = 2
_N_ROPES = 2
_N_CONVEYORS = 1
_N_PLATFORMS = 12
_N_GAME = (
    _N_ENEMIES
    + _N_WAYPOINTS
    + _N_ITEMS
    + _N_DOORS
    + _N_ROPES
    + _N_CONVEYORS
    + _N_PLATFORMS
)  # 31
_MONTEZUMA_MAX_OBJECTS = _N_GAME + 1  # + meta NOOP

# Standing player height in jax_montezumarevenge; waypoint y = floor - half.
_PLAYER_STAND_HEIGHT = 20
_PLAYER_STAND_HALF = _PLAYER_STAND_HEIGHT // 2
# Climb shaft centre is ladders_x + 8 (see jax_montezumarevenge climb snap).
_LADDER_SHAFT_HALF = 8
# Raise top waypoints so arrive does not fire mid-climb.
_TOP_WAYPOINT_LIFT_PX = 12
# Kangaroo-style readable waypoint footprint (was 1×1 — invisible in dashboard).
_WAYPOINT_W = 8
_WAYPOINT_H = 8
_WAYPOINT_HALF_W = _WAYPOINT_W // 2
# Top WPs within this centre distance of the player are treated as already
# reached (spawn sits on the top floor next to the center ladder).
_TOP_WP_ALREADY_THERE_PX = 16.0
# jaxatari draws room content at state_y + 47 (HUD band). Observations use the
# same offset so overlays / ego match the rendered frame.
MONTEZUMA_RENDER_Y_OFFSET = 47

# Game-layout indices (before meta-NOOP). Policy slots = these + 1.
WP_BOT0 = _N_ENEMIES  # 3
WP_TOP0 = _N_ENEMIES + _N_LADDERS  # 7
ITEM0 = _N_ENEMIES + _N_WAYPOINTS  # 11
DOOR0 = ITEM0 + _N_ITEMS  # 14
ROPE0 = DOOR0 + _N_DOORS  # 16
PLATFORM0 = DOOR0 + _N_DOORS + _N_ROPES + _N_CONVEYORS  # 19

# High rope-grab COLLECT: top of shaft, above mid-floor stand height so the
# skill must jump the chasm (walking under the rope does not arrive).
_ROPE_GRAB_W = 8
_ROPE_GRAB_H = 8
_ROPE_GRAB_TOP_INSET = 2  # px below ropes_top

# Room-4 mid-tier scaffolding / gap geometry lives in montezuma_policy
# (numbered WP* centres).  This wrapper only stamps those obs-space boxes.


def _squeeze_trailing_singleton(leaf):
    """Collapse a trailing size-1 axis (jaxatari player leaves as length-1 vectors)."""
    leaf = jnp.asarray(leaf)
    if leaf.ndim >= 1 and leaf.shape[-1] == 1:
        return jnp.squeeze(leaf, axis=-1)
    return leaf


def _orientation_from_player_dir(player_dir):
    """Map Montezuma ``player_dir`` ±1 → Kangaroo-style degrees (right=90, left=270)."""
    return jnp.where(
        jnp.asarray(player_dir) == 1,
        jnp.float32(90.0),
        jnp.float32(270.0),
    )


def _ladder_waypoints_from_state(env_state, player=None):
    """Build bottom/top COLLECT waypoints from ``env_state.ladders_*``.

    Room-1 path uses center + both side ladders (right down, left up to key).
    Coordinates are in *state* space; ``_patch_raw_observation`` adds the
    render Y offset.

    The center TOP waypoint is cleared while the player is already on that
    landing (spawn), and parked off-screen so it cannot look like an
    unfinished checkpoint the agent is standing inside.
    """
    lad_cx = env_state.ladders_x.astype(jnp.float32) + float(_LADDER_SHAFT_HALF)
    active = env_state.ladders_active.astype(jnp.int32)
    bot_y = env_state.ladders_bottom.astype(jnp.float32) - float(_PLAYER_STAND_HALF)
    top_y = (
        env_state.ladders_top.astype(jnp.float32)
        - float(_PLAYER_STAND_HALF)
        - float(_TOP_WAYPOINT_LIFT_PX)
    )
    bot_x = lad_cx - float(_WAYPOINT_HALF_W)
    top_x = lad_cx - float(_WAYPOINT_HALF_W)
    w = jnp.full_like(lad_cx, float(_WAYPOINT_W))
    h = jnp.full_like(lad_cx, float(_WAYPOINT_H))

    bot_active = active
    top_active = active

    if player is not None:
        px = jnp.asarray(player.x, dtype=jnp.float32)
        py = jnp.asarray(player.y, dtype=jnp.float32)
        pw = jnp.asarray(player.width, dtype=jnp.float32)
        ph = jnp.asarray(player.height, dtype=jnp.float32)
        px = px[-1] if px.ndim > 0 else px
        py = py[-1] if py.ndim > 0 else py
        pw = pw[-1] if pw.ndim > 0 else pw
        ph = ph[-1] if ph.ndim > 0 else ph
        # Called before render-Y offset is applied — player and ladders share state-y.
        p_cx = px + pw * 0.5
        p_cy = py + ph * 0.5
        top_cx = top_x + float(_WAYPOINT_HALF_W)
        top_cy = top_y + float(_WAYPOINT_H) * 0.5
        dist = jnp.hypot(p_cx - top_cx, p_cy - top_cy)
        # Clear only while already on the upper landing — not while climbing up.
        on_upper = p_cy <= (
            env_state.ladders_top.astype(jnp.float32) + jnp.float32(6.0)
        )
        already_there = jnp.logical_and(
            dist <= jnp.float32(_TOP_WP_ALREADY_THERE_PX), on_upper
        )
        top_active = jnp.where(
            already_there, jnp.zeros_like(top_active), top_active
        ).astype(jnp.int32)

    # Hide inactive tops so spawn does not sit inside a visible dead waypoint.
    top_x = jnp.where(top_active > 0, top_x, jnp.zeros_like(top_x))
    top_y = jnp.where(top_active > 0, top_y, jnp.full_like(top_y, -40.0))

    return {
        "bot_x": bot_x,
        "bot_y": bot_y,
        "top_x": top_x,
        "top_y": top_y,
        "w": w,
        "h": h,
        "bot_active": bot_active,
        "top_active": top_active,
    }


class MontezumaRevengeSemanticWrapper(JaxatariSemanticBaseWrapper):
    def __init__(self, env):
        super().__init__(env, max_objects=_MONTEZUMA_MAX_OBJECTS)

    def _patch_raw_observation(self, semantic_obs, env_state):
        """Squeeze player leaves, stamp facing/death, inject WPs, align Y to render."""
        unwrapped = self._unwrap_env_state(env_state.atari_state)

        player = semantic_obs.raw_observation.player
        new_player = player.replace(
            x=_squeeze_trailing_singleton(player.x),
            y=_squeeze_trailing_singleton(player.y),
            width=_squeeze_trailing_singleton(player.width),
            height=_squeeze_trailing_singleton(player.height),
            active=_squeeze_trailing_singleton(player.active),
            orientation=_squeeze_trailing_singleton(player.orientation),
            state=_squeeze_trailing_singleton(player.state),
        )
        ori = _orientation_from_player_dir(unwrapped.player_dir)
        cur_ori = new_player.orientation
        if cur_ori.ndim == 0:
            new_ori = ori.astype(jnp.float32)
        else:
            new_ori = jnp.asarray(cur_ori, dtype=jnp.float32).at[-1].set(ori)
        dying = jnp.asarray(unwrapped.death_timer > 0, dtype=jnp.bool_)
        death_state = jnp.where(
            dying, jnp.int32(PLAYER_STATE_DYING), jnp.int32(0)
        )
        cur_state = jnp.asarray(new_player.state, dtype=jnp.int32)
        if cur_state.ndim == 0:
            new_state = death_state
        else:
            new_state = cur_state.at[-1].set(death_state)
        new_player = new_player.replace(orientation=new_ori, state=new_state)
        semantic_obs = semantic_obs.replace(
            raw_observation=semantic_obs.raw_observation.replace(player=new_player)
        )
        # Inject in state-y, then shift player + objects to match renderer (y+47).
        semantic_obs = self._inject_ladder_waypoints(semantic_obs, unwrapped, new_player)
        semantic_obs = self._inject_rope_grab_collects(semantic_obs, unwrapped)
        semantic_obs = self._apply_render_y_offset(semantic_obs)
        # Path WP centres live in obs space (see montezuma_policy.WP*); stamp last.
        return self._apply_room4_path_stamps(semantic_obs, unwrapped)

    def _apply_render_y_offset(self, semantic_obs):
        """Map state-y observations onto the rendered room (HUD band is 47px)."""
        off = jnp.int32(MONTEZUMA_RENDER_Y_OFFSET)
        player = semantic_obs.raw_observation.player
        py = jnp.asarray(player.y)
        new_player = player.replace(y=(py + off).astype(py.dtype))
        objects = semantic_obs.objects
        oy = jnp.asarray(objects.y)
        new_objects = objects.replace(y=(oy + off).astype(oy.dtype))
        return semantic_obs.replace(
            raw_observation=semantic_obs.raw_observation.replace(player=new_player),
            objects=new_objects,
        )

    def _inject_ladder_waypoints(self, semantic_obs, env_state, player):
        """Fill reserved COLLECT waypoint slots from current room ladders."""
        wp = _ladder_waypoints_from_state(env_state, player=player)
        objects = semantic_obs.objects
        # Object arrays are [stack, max_objects] after meta prepend.
        # Game layout starts at meta_reserved_slots (usually 1).
        meta = self.meta_reserved_slots
        bot0 = meta + WP_BOT0
        top0 = meta + WP_TOP0
        n = _N_LADDERS

        def _set_block(leaf, start, values, cast=True):
            vals = jnp.asarray(values)
            if cast:
                vals = vals.astype(leaf.dtype)
            # Broadcast across frame stack: write identical waypoints on every frame.
            stacked = jnp.broadcast_to(vals, (leaf.shape[0], n))
            return leaf.at[:, start : start + n].set(stacked)

        new_x = _set_block(objects.x, bot0, wp["bot_x"])
        new_x = _set_block(new_x, top0, wp["top_x"])
        new_y = _set_block(objects.y, bot0, wp["bot_y"])
        new_y = _set_block(new_y, top0, wp["top_y"])
        new_w = _set_block(objects.width, bot0, wp["w"])
        new_w = _set_block(new_w, top0, wp["w"])
        new_h = _set_block(objects.height, bot0, wp["h"])
        new_h = _set_block(new_h, top0, wp["h"])
        new_act = _set_block(objects.active, bot0, wp["bot_active"])
        new_act = _set_block(new_act, top0, wp["top_active"])

        labels = jnp.full(n, LABEL_COLLECT, dtype=objects.semantic_label.dtype)
        new_lab = objects.semantic_label.at[:, bot0 : bot0 + n].set(
            jnp.broadcast_to(labels, (objects.semantic_label.shape[0], n))
        )
        new_lab = new_lab.at[:, top0 : top0 + n].set(
            jnp.broadcast_to(labels, (new_lab.shape[0], n))
        )

        return semantic_obs.replace(
            objects=objects.replace(
                x=new_x,
                y=new_y,
                width=new_w,
                height=new_h,
                active=new_act,
                semantic_label=new_lab,
            )
        )

    def _inject_rope_grab_collects(self, semantic_obs, env_state):
        """Replace tall rope AABBs with a high COLLECT grab point.

        The native rope box spans top→mid, so a standing player on the mid
        floor already overlaps it. A small WP near ``ropes_top`` sits above
        stand height — COLLECT must jump the chasm to arrive (room-1 right gap).
        """
        objects = semantic_obs.objects
        meta = self.meta_reserved_slots
        rope0 = meta + ROPE0
        n = _N_ROPES
        active = env_state.ropes_active.astype(jnp.int32)
        # Centre the grab box on the rope shaft.
        x = env_state.ropes_x.astype(jnp.float32) - float(_ROPE_GRAB_W) * 0.5
        y = (
            env_state.ropes_top.astype(jnp.float32)
            + float(_ROPE_GRAB_TOP_INSET)
        )
        w = jnp.full((n,), float(_ROPE_GRAB_W))
        h = jnp.full((n,), float(_ROPE_GRAB_H))
        # Park inactive ropes off-screen.
        x = jnp.where(active > 0, x, jnp.zeros_like(x))
        y = jnp.where(active > 0, y, jnp.full_like(y, -40.0))
        labs = jnp.full(n, LABEL_COLLECT, dtype=objects.semantic_label.dtype)

        def _set(leaf, values):
            vals = jnp.asarray(values).astype(leaf.dtype)
            stacked = jnp.broadcast_to(vals, (leaf.shape[0], n))
            return leaf.at[:, rope0 : rope0 + n].set(stacked)

        return semantic_obs.replace(
            objects=objects.replace(
                x=_set(objects.x, x),
                y=_set(objects.y, y),
                width=_set(objects.width, w),
                height=_set(objects.height, h),
                active=_set(objects.active, active),
                semantic_label=_set(objects.semantic_label, labs),
            )
        )

    def _apply_room4_path_stamps(self, semantic_obs, env_state):
        """Overwrite room-4 path WPs from ``montezuma_policy`` WP* centres.

        Late-import avoids a circular import (policy needs PLAYER_STATE_DYING).
        Ladder tops that the climb injector already marked inactive (spawn on
        L0_TOP) keep ``active=0``; scaffolding slots are forced active.
        """
        # Late import: policy imports PLAYER_STATE_DYING from this module.
        from concept_based_learning.environments.montezumarevenge.montezuma_policy import (
            _SLOT_EDGE,
            _SLOT_GAP_L,
            _SLOT_GAP_R,
            _SLOT_ROPE,
            _SLOT_SHAFT,
            room4_path_stamps,
        )

        objects = semantic_obs.objects
        in_room4 = jnp.asarray(env_state.room_id, dtype=jnp.int32) == jnp.int32(4)
        stamps = room4_path_stamps()
        force_active = {_SLOT_SHAFT, _SLOT_EDGE, _SLOT_ROPE}
        force_inactive = {_SLOT_GAP_L, _SLOT_GAP_R}

        new_x = objects.x
        new_y = objects.y
        new_w = objects.width
        new_h = objects.height
        new_act = objects.active
        new_lab = objects.semantic_label

        for slot, (x, y, w, h, lab) in stamps.items():
            def _set1(leaf, value, _slot=slot):
                vals = jnp.asarray(value).astype(leaf.dtype)
                stacked = jnp.broadcast_to(vals, (leaf.shape[0],))
                return leaf.at[:, _slot].set(stacked)

            # Park off-screen when not in room 4.
            sx = jnp.where(in_room4, jnp.float32(x), jnp.float32(0.0))
            sy = jnp.where(in_room4, jnp.float32(y), jnp.float32(-40.0))
            new_x = _set1(new_x, sx)
            new_y = _set1(new_y, sy)
            new_w = _set1(new_w, float(w))
            new_h = _set1(new_h, float(h))
            new_lab = _set1(new_lab, int(lab))
            if slot in force_active:
                new_act = _set1(
                    new_act,
                    jnp.where(in_room4, jnp.int32(1), jnp.int32(0)),
                )
            elif slot in force_inactive:
                new_act = _set1(new_act, jnp.int32(0))
            # else: keep ladder active flags from _inject_ladder_waypoints

        return semantic_obs.replace(
            objects=objects.replace(
                x=new_x,
                y=new_y,
                width=new_w,
                height=new_h,
                active=new_act,
                semantic_label=new_lab,
            )
        )

    def reset(self, key=None):
        semantic_obs, state = super().reset(key)
        return self._patch_raw_observation(semantic_obs, state), state

    def step(self, state, action):
        semantic_obs, new_state, reward, done, info = super().step(state, action)
        semantic_obs = self._patch_raw_observation(semantic_obs, new_state)
        return semantic_obs, new_state, reward, done, info

    def _substep_collect_events(self, prev_env_state, next_env_state):
        """Item pickup → was_collected (items_active 1→0 without life loss)."""
        life_lost = next_env_state.lives < prev_env_state.lives
        item_died = jnp.logical_and(
            prev_env_state.items_active == 1,
            next_env_state.items_active == 0,
        )
        collected = jnp.logical_and(item_died, jnp.logical_not(life_lost)).astype(
            jnp.int32
        )
        # Slot layout: enemies | waypoints | items | doors | ropes | conveyors | platforms
        zeros_before = jnp.zeros(_N_ENEMIES + _N_WAYPOINTS, dtype=jnp.int32)
        zeros_after = jnp.zeros(
            _N_DOORS + _N_ROPES + _N_CONVEYORS + _N_PLATFORMS, dtype=jnp.int32
        )
        return jnp.concatenate([zeros_before, collected, zeros_after])

    def _stamp_collect_events_on_observation(self, semantic_obs, collect_mask):
        if collect_mask is None:
            return semantic_obs
        was_collected_with_noop = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), collect_mask.astype(jnp.int32)]
        )
        objects = semantic_obs.objects
        new_wc = objects.was_collected.at[-1].set(was_collected_with_noop)
        return semantic_obs.replace(objects=objects.replace(was_collected=new_wc))

    def _extract_frame_objects(self, obs) -> LabeledObjectObservation:
        enemies = obs.enemies
        items = obs.items
        doors = obs.doors
        ropes = obs.ropes
        conveyors = obs.conveyors
        platforms = obs.platforms

        # Ladder waypoints: reserved inactive slots; filled from env_state in patch.
        wp_zeros = jnp.zeros(_N_LADDERS, dtype=enemies.x.dtype)
        wp_ones = jnp.ones(_N_LADDERS, dtype=enemies.x.dtype)
        wp_inactive = jnp.zeros(_N_LADDERS, dtype=jnp.int32)

        all_x = jnp.concatenate(
            [
                enemies.x,
                wp_zeros,
                wp_zeros,
                items.x,
                doors.x,
                ropes.x,
                conveyors.x,
                platforms.x,
            ]
        )
        all_y = jnp.concatenate(
            [
                enemies.y,
                wp_zeros,
                wp_zeros,
                items.y,
                doors.y,
                ropes.y,
                conveyors.y,
                platforms.y,
            ]
        )
        all_w = jnp.concatenate(
            [
                enemies.width,
                wp_ones,
                wp_ones,
                items.width,
                doors.width,
                ropes.width,
                conveyors.width,
                platforms.width,
            ]
        )
        all_h = jnp.concatenate(
            [
                enemies.height,
                wp_ones,
                wp_ones,
                items.height,
                doors.height,
                ropes.height,
                conveyors.height,
                platforms.height,
            ]
        )
        all_active = jnp.concatenate(
            [
                enemies.active.astype(jnp.int32),
                wp_inactive,
                wp_inactive,
                items.active.astype(jnp.int32),
                doors.active.astype(jnp.int32),
                ropes.active.astype(jnp.int32),
                conveyors.active.astype(jnp.int32),
                platforms.active.astype(jnp.int32),
            ]
        )
        all_labels = jnp.concatenate(
            [
                jnp.full(_N_ENEMIES, LABEL_AVOID, dtype=jnp.int32),
                jnp.full(_N_WAYPOINTS, LABEL_COLLECT, dtype=jnp.int32),
                jnp.full(_N_ITEMS, LABEL_COLLECT, dtype=jnp.int32),
                jnp.full(_N_DOORS, LABEL_COLLECT, dtype=jnp.int32),
                jnp.full(_N_ROPES, LABEL_COLLECT, dtype=jnp.int32),
                jnp.full(_N_CONVEYORS, LABEL_BLOCKER, dtype=jnp.int32),
                jnp.full(_N_PLATFORMS, LABEL_BLOCKER, dtype=jnp.int32),
            ]
        )
        n_objects = all_x.shape[0]
        pad_size = self.game_max_objects - n_objects
        pad_x = jnp.zeros(pad_size, dtype=all_x.dtype)
        pad_y = jnp.zeros(pad_size, dtype=all_y.dtype)
        pad_w = jnp.zeros(pad_size, dtype=all_w.dtype)
        pad_h = jnp.zeros(pad_size, dtype=all_h.dtype)
        pad_active = jnp.zeros(pad_size, dtype=jnp.int32)
        pad_labels = jnp.full(pad_size, LABEL_NOOP, dtype=jnp.int32)
        pad_offscreen = jnp.zeros(pad_size, dtype=jnp.int32)

        return LabeledObjectObservation(
            x=jnp.concatenate([all_x, pad_x]),
            y=jnp.concatenate([all_y, pad_y]),
            width=jnp.concatenate([all_w, pad_w]),
            height=jnp.concatenate([all_h, pad_h]),
            active=jnp.concatenate([all_active, pad_active]),
            semantic_label=jnp.concatenate([all_labels, pad_labels]),
            state=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
            can_go_offscreen=jnp.concatenate(
                [jnp.zeros(n_objects, dtype=jnp.int32), pad_offscreen]
            ),
            was_killed=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
            was_collected=jnp.zeros(n_objects + pad_size, dtype=jnp.int32),
        )
