import functools
import jax
import jax.numpy as jnp
from jaxatari.wrappers import AtariWrapper, JaxatariWrapper
import jaxatari.spaces as spaces
from jaxatari.games import jax_seaquest as jaxatari_seaquest
from typing import Optional
from flax import struct
import chex

from src.environments.seaquest.datatypes import ConceptObs
import src.environments.seaquest.seaquest_constants as const
from src.config import Config, get_default_config

class SeaquestConceptWrapper(JaxatariWrapper):
    """
    Neurosymbolic Interface for Seaquest.
    1. Extracts Objects & Globals.
    2. LABELS objects (Semantic Classes).
    3. DETECTS EVENTS (Symbolic Ground Truth for Rewards).
    """
    def __init__(self, env, config: Optional[Config] = None):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "Must wrap AtariWrapper first"
        
        # Use provided config or default
        if config is None:
            config = get_default_config()
        self.config = config
        
        self._env = env
        self.stack_size = self._env.frame_stack_size
        self.SCREEN_WIDTH = config.environment.screen_width
        self.SCREEN_HEIGHT = config.environment.screen_height

        # Object Capacity (Total: 34, Player Missile removed)
        # 0-11: Sharks (Avoid)
        # 12-23: Subs (Avoid)
        # 24-27: Divers (Collect)
        # 28-31: Enemy Missiles (Avoid)
        # 32: Surface Sub (Monitor)
        # Player Missile is no longer in object list - handled via capabilities
        self.max_objs = config.environment.max_objs
        
        self.n_global_feats = config.environment.n_global_feats
        # TODO: maybe append the label to the object features as well, so they are part of the attention mechanism in meta policy and instance selector
        self.n_obj_feats = config.environment.n_obj_feats  # x, y, w, h

        self._observation_space = self._make_observation_space()

    def _make_observation_space(self) -> spaces.Dict:
        num_concepts = self.config.concepts.label_high_level_concept + 1
        return spaces.Dict({
            # Dashboard (Oxygen, Ammo, etc.)
            "global_features": spaces.Box(0.0, 2.0, (self.n_global_feats,), jnp.float32),
            
            # Capabilities
            "capabilities": spaces.Box(0.0, 1.0, (const.N_CAPABILITIES,), jnp.float32),
            
            # Object History [35, Stack*4] -> Flattened for network
            "objects": spaces.Box(0.0, 1.0, (self.max_objs, self.stack_size * self.n_obj_feats), jnp.float32),
            
            # Existence Mask
            "object_mask": spaces.Box(0.0, 1.0, (self.max_objs,), jnp.float32),
            
            # Semantic Labels
            "object_labels": spaces.Box(0, 1, (self.max_objs, num_concepts), jnp.float32),
            
            # Symbolic Event Channel (0=None, 1=Interaction)
            "object_events": spaces.Box(0, 2, (self.max_objs,), jnp.int32),
            
            # Internal placeholders
            "active_concept": spaces.Box(0, 10, (), jnp.int32),
            "active_target_idx": spaces.Box(0, self.max_objs, (), jnp.int32)
        })

    def observation_space(self) -> spaces.Dict:
        return self._observation_space

    # --- 1. Event Oracle (The Logic Shift) ---
    def _detect_events(self, prev_state, curr_state, player_y_norm):
        """
        Generates symbolic event flags for the reward function (like flagging a diver as collected).
        This enables the Reward Function to be 100% generic.
        
        Args:
            prev_state: Previous environment state
            curr_state: Current environment state
            player_y_norm: Normalized player Y position (from state/observation)
        """
        events = jnp.zeros(self.max_objs, dtype=jnp.int32)

        # Extract SeaquestState from AtariState wrapper
        prev_seaquest = prev_state.env_state
        curr_seaquest = curr_state.env_state

        # A. COLLECTION EVENTS (Divers: Indices 24-27)
        # Logic: Score up + Diver disappeared
        score_up = (curr_seaquest.score > prev_seaquest.score) # Generic score check
        # Note: Seaquest specific: divers_collected count increases
        divers_increased = (curr_seaquest.divers_collected > prev_seaquest.divers_collected)
        
        # Check active status change for divers
        # Position arrays are (4, 3): [x, y, active_or_direction]
        # Index 2 is the active flag (0 = inactive, non-zero = active)
        d_prev_act = prev_seaquest.diver_positions[:, 2] != 0
        d_curr_act = curr_seaquest.diver_positions[:, 2] != 0
        d_disappeared = jnp.logical_and(d_prev_act, ~d_curr_act)
        
        # Tag specific diver
        d_event = jnp.logical_and(d_disappeared, divers_increased)
        events = events.at[24:28].set(d_event.astype(jnp.int32))

        # B. DESTRUCTION EVENTS (Enemies: Sharks 0-11, Subs 12-23)
        # Logic: Score up + Enemy disappeared (and NOT a diver collection)
        # This prevents awarding "Destroy" when we actually "Collected"
        combat_score = jnp.logical_and(score_up, ~divers_increased)
        
        # Sharks - position arrays are (12, 3): [x, y, active_or_direction]
        s_prev_act = prev_seaquest.shark_positions[:, 2] != 0
        s_curr_act = curr_seaquest.shark_positions[:, 2] != 0
        s_gone = jnp.logical_and(s_prev_act, ~s_curr_act)
        events = events.at[0:12].set(jnp.logical_and(s_gone, combat_score).astype(jnp.int32))
        
        # Subs - position arrays are (12, 3): [x, y, active_or_direction]
        sub_prev_act = prev_seaquest.sub_positions[:, 2] != 0
        sub_curr_act = curr_seaquest.sub_positions[:, 2] != 0
        sub_gone = jnp.logical_and(sub_prev_act, ~sub_curr_act)
        events = events.at[12:24].set(jnp.logical_and(sub_gone, combat_score).astype(jnp.int32))

        # C. COLLISION / AVOID [FIXED]
        # Old: lives < prev_lives (Too slow, waits for animation)
        # New: death_counter just activated (0 -> >0)
        # Note: We verify lives didn't INCREASE (just in case of 10k point bonus)
        just_died = jnp.logical_and(prev_seaquest.death_counter == 0, curr_seaquest.death_counter > 0)
        
        # If we died, tag all active enemies as "Danger"
        if_hit_tag_sharks = jnp.logical_and(just_died, s_prev_act)
        if_hit_tag_subs = jnp.logical_and(just_died, sub_prev_act)
        
        # Check enemy missiles (indices 28-31)
        m_prev_act = prev_seaquest.enemy_missile_positions[:, 2] != 0
        if_hit_tag_missiles = jnp.logical_and(just_died, m_prev_act)

        # Apply tags
        events = events.at[0:12].max(if_hit_tag_sharks.astype(jnp.int32))
        events = events.at[12:24].max(if_hit_tag_subs.astype(jnp.int32))
        events = events.at[28:32].max(if_hit_tag_missiles.astype(jnp.int32))

        # D. MONITOR/SURFACE EVENT (Oxygen Refill)
        # Logic: Oxygen increased + Player is at the top + Oxygen not full
        refilled = curr_seaquest.oxygen > prev_seaquest.oxygen
        
        # Ignore refill if we are in "Init State" (just_surfaced == -1) or if we are dead/dying (death_counter > 0)
        is_playing = jnp.logical_and(curr_seaquest.just_surfaced != -1, curr_seaquest.death_counter == 0)
        
        # Check if player is at the top (Y=0 is top in Seaquest, normalized Y should be small)
        # Using threshold of 0.1 (10% from top) to account for player height
        # player_y_norm is passed from step() where it's extracted from the observation
        is_at_top = player_y_norm < 0.1
        
        # Check if oxygen is not full (max oxygen is 64)
        oxygen_not_full = curr_seaquest.oxygen < 64
        
        # Combine all conditions: refilling, playing, at top, and not full
        valid_refill = jnp.logical_and(refilled, is_playing)
        valid_refill = jnp.logical_and(valid_refill, is_at_top)
        valid_refill = jnp.logical_and(valid_refill, oxygen_not_full)
        
        events = events.at[32].set(valid_refill.astype(jnp.int32))

        return events

    # --- 2. Normalization & Stacking ---
    def _normalize_feats(self, entity_array, label_array):
        # Input: [N, 5] (x, y, w, h, active)
        # Output: feats[N, 4], mask[N], labels[N]
        x, y, w, h, active = entity_array[:, 0], entity_array[:, 1], entity_array[:, 2], entity_array[:, 3], entity_array[:, 4]
        
        # Norm [0, 1]
        feats = jnp.stack([
            x / self.SCREEN_WIDTH, 
            y / self.SCREEN_HEIGHT, 
            w / self.SCREEN_WIDTH, 
            h / self.SCREEN_HEIGHT
        ], axis=-1)
        
        
        # create multi-hot encoding of labels
        num_concepts = self.config.concepts.label_high_level_concept + 1
        
        def make_one_hot(idx):
            return jax.nn.one_hot(idx, num_classes = num_concepts)

        # sum over the labels to get the multi-hot encoding
        multi_hot = jnp.sum(jax.vmap(make_one_hot)(label_array), axis=0)

        # broadcast multi-hot encoding to the shape of the entitiy array
        labels = jnp.tile(multi_hot, (entity_array.shape[0], 1))

        mask = active.astype(jnp.float32)
        return feats, mask, labels

    def _process_single_frame(self, obs, state):
        """Extracts features from ONE frame."""
        # Helper for scalars
        def sc(e): 
            return jnp.array([[e.x, e.y, e.width, e.height, e.active]])

        # Flax structs should support direct attribute access even in JAX transformations
        # If this fails, the struct needs to be properly registered with JAX
        # 1. Process Groups
        
        # Enemies -> DESTROY (or AVOID, depending on strategy. Let's use DESTROY as primary for enemies we can kill)
        sh_ids = jnp.array([const.LABEL_DESTROY, const.LABEL_AVOID])
        sh_f, sh_m, sh_l = self._normalize_feats(obs.sharks, sh_ids)
        su_f, su_m, su_l = self._normalize_feats(obs.submarines, sh_ids)
        
        # Divers -> COLLECT
        di_ids = jnp.array([const.LABEL_COLLECT])
        di_f, di_m, di_l = self._normalize_feats(obs.divers, di_ids)
        
        # Enemy Missiles -> AVOID (Cannot destroy them usually)
        em_ids = jnp.array([const.LABEL_AVOID])
        em_f, em_m, em_l = self._normalize_feats(obs.enemy_missiles, em_ids)
        
        # Surface Sub -> AVOID
        ss_ids = jnp.array([const.LABEL_AVOID])
        ss_f, ss_m, ss_l = self._normalize_feats(sc(obs.surface_submarine), ss_ids)
        
        # Player Missile is no longer in the object list - it's handled via capabilities (can_fire)

        # 2. Concat (Removed pm_f, pm_m, pm_l)
        # Actual objects: 12 sharks + 12 subs + 4 divers + 4 enemy missiles + 1 surface sub = 33
        # But max_objs is 34, so we need to pad with 1 zero object
        objects_raw = jnp.concatenate([sh_f, su_f, di_f, em_f, ss_f], axis=0)
        mask_raw = jnp.concatenate([sh_m, su_m, di_m, em_m, ss_m], axis=0)
        labels_raw = jnp.concatenate([sh_l, su_l, di_l, em_l, ss_l], axis=0)
        
        # Pad to max_objs (always pad, JAX needs static shapes)
        n_actual = objects_raw.shape[0]
        pad_size = self.max_objs - n_actual
        objects = jnp.concatenate([objects_raw, jnp.zeros((pad_size, objects_raw.shape[1]))], axis=0)
        mask = jnp.concatenate([mask_raw, jnp.zeros(pad_size)], axis=0)
        labels = jnp.concatenate([labels_raw, jnp.zeros((pad_size, labels_raw.shape[1]))], axis=0)

        # 3. Global Features (Includes Player Position)
        oxy_norm = obs.oxygen_level / 64.0
        divers_held = obs.collected_divers / 6.0
        lives_norm = obs.lives / 3.0
        
        # Calculate simple densities
        enemy_count = jnp.sum(sh_m) + jnp.sum(su_m)
        diver_count = jnp.sum(di_m)
        
        player_width = obs.player.width / self.SCREEN_WIDTH
        player_height = obs.player.height / self.SCREEN_HEIGHT
        # Normalize orientation from [-1, 1] to [0, 1]: -1 -> 0.0, 1 -> 1.0
        player_orientation = (obs.player.o + 1.0) / 2.0

        global_features = jnp.stack([
            oxy_norm, divers_held, lives_norm, 
            enemy_count, diver_count,
            obs.player.x / self.SCREEN_WIDTH,
            obs.player.y / self.SCREEN_HEIGHT,
            player_width, player_height, player_orientation
        ])

        # Extract missile active status for capabilities
        missile_active = obs.player_missile.active

        return objects, mask, labels, global_features, missile_active

    def _orientation_to_facing_vec(self, orientation):
        """
        Convert player orientation to facing vector.
        In Seaquest: orientation 1 = Right (facing [1, 0]), orientation -1 = Left (facing [-1, 0])
        """
        # orientation: 1 = right, -1 = left
        # Use orientation directly as facing_x (1 for right, -1 for left)
        facing_x = orientation.astype(jnp.float32)  # 1.0 for right, -1.0 for left
        facing_y = 0.0  # Seaquest is horizontal only
        return jnp.array([facing_x, facing_y])
    

    def _process_stack(self, obs_stack, events):
        """
        Flattens history, attaches events.
        """
        # VMAP over stack to get features [Stack, N, 4]
        # We need to pass a dummy state or just use the obs, but _process_single_frame only needs obs.
        # Wait, _process_single_frame calls obs.sharks...
        objects_stack, mask_stack, labels_stack, globals_stack, missile_active_stack = jax.vmap(
            lambda o: self._process_single_frame(o, None) # State unused in feature extraction
        )(obs_stack)

        # Flatten Objects: [Stack, N, 4] -> [N, Stack*4]
        # Transpose to [N, Stack, 4]
        n_objs = objects_stack.shape[1]
        objs_T = jnp.transpose(objects_stack, (1, 0, 2))
        flat_objects = objs_T.reshape(n_objs, -1)

        # Take Latest Globals/Masks
        latest_globals = globals_stack[-1]
        latest_mask = mask_stack[-1]
        latest_labels = labels_stack[-1]

        # --- Compute Capabilities (UPDATED) ---
        # Extract missile status from latest frame (last element of processed stack)
        # Seaquest logic: If missile is active, you cannot fire
        latest_missile_active = missile_active_stack[-1]
        missile_active = latest_missile_active > 0.5
        can_fire = jnp.where(missile_active, 0.0, 1.0)
        
        caps = jnp.zeros(const.N_CAPABILITIES)
        caps = caps.at[const.CAP_CAN_FIRE].set(can_fire)
        
        # 2. Inject Dynamic Shooting Vector (from player orientation)
        # Extract orientation from global_features (index 9) and denormalize from [0, 1] back to [-1, 1]
        # Normalized: 0.0 = -1 (left), 1.0 = 1 (right)
        player_orientation_norm = latest_globals[9]
        player_orientation = player_orientation_norm * 2.0 - 1.0  # Denormalize: 0.0 -> -1.0, 1.0 -> 1.0
        facing_vec = self._orientation_to_facing_vec(player_orientation)
        caps = caps.at[const.CAP_FACING_X].set(facing_vec[0])
        caps = caps.at[const.CAP_FACING_Y].set(facing_vec[1])

        # Events should always be a JAX array (zeros on reset, detected events on step)
        # No need for None check - always a valid JAX array

        return ConceptObs(
            global_features=latest_globals,
            capabilities=caps,
            objects=flat_objects,
            object_mask=latest_mask,
            object_labels=latest_labels,
            object_events=events,
            active_concept=jnp.array(0, dtype=jnp.int32),
            active_target_idx=jnp.array(0, dtype=jnp.int32)
        )

    # --- 3. Main Loop Overrides ---
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        obs_stack, env_state = self._env.reset(key)
        zero_events = jnp.zeros(self.max_objs, dtype=jnp.int32)
        return self._process_stack(obs_stack, zero_events), env_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # 1. Capture Previous State (for Event Detection)
        prev_state = state
        
        # 2. Step
        obs_stack, next_state, reward, done, info = self._env.step(state, action)
        
        # 3. Detect Events (The Oracle)
        # Extract player Y position from the current observation (latest frame)
        # The observation comes from state, so this is effectively using state information
        # Process a single frame to extract player Y in a JAX-compatible way
        # obs_stack is a tuple-like object (Flax struct); extract last element using tree_leaves
        # obs_stack acts like a tuple, so we can use indexing if it's not a tracer
        # For JAX compatibility, process the entire obs_stack and take the last result
        # Since _process_stack uses vmap, we can process obs_stack and get the last frame's features
        _, _, _, global_features_temp, _ = jax.vmap(
            lambda o: self._process_single_frame(o, None)
        )(obs_stack)
        global_features = global_features_temp[-1]
        # Player Y is at index 6 in global_features (normalized)
        player_y_norm = global_features[6]
        events = self._detect_events(prev_state, next_state, player_y_norm)
        
        # 4. Process
        concept_obs = self._process_stack(obs_stack, events)
        
        return concept_obs, next_state, reward, done, info