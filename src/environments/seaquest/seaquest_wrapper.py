import functools
import jax
import jax.numpy as jnp
from jaxatari.wrappers import AtariWrapper, JaxatariWrapper
import jaxatari.spaces as spaces
from jaxatari.games import jax_seaquest as jaxatari_seaquest

from src.environments.seaquest.datatypes import ConceptObs
import src.environments.seaquest.seaquest_constants as const

class SeaquestConceptWrapper(JaxatariWrapper):
    """
    Neurosymbolic Interface for Seaquest.
    1. Extracts Objects & Globals.
    2. LABELS objects (Semantic Classes).
    3. DETECTS EVENTS (Symbolic Ground Truth for Rewards).
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "Must wrap AtariWrapper first"
        
        self._env = env
        self.stack_size = self._env.frame_stack_size
        self.SCREEN_WIDTH = 160.0
        self.SCREEN_HEIGHT = 210.0

        # Object Capacity (Total: 35)
        # 0: Player
        # 1-12: Sharks (Avoid)
        # 13-24: Subs (Avoid)
        # 25-28: Divers (Collect)
        # 29-32: Enemy Missiles (Avoid)
        # 33: Surface Sub (Monitor)
        # 34: Player Missile (Player Tool)
        self.max_objs = 35 
        
        self.n_global_feats = 10 
        self.n_obj_feats = 4 # x, y, w, h

        # Define ranges for easy checking
        self.idx_player_missile = 34

        self._observation_space = self._make_observation_space()

    def _make_observation_space(self) -> spaces.Dict:
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
            "object_labels": spaces.Box(0, 10, (self.max_objs,), jnp.int32),
            
            # NEW: Symbolic Event Channel (0=None, 1=Interaction)
            "object_events": spaces.Box(0, 2, (self.max_objs,), jnp.int32),
            
            # Internal placeholders
            "active_concept": spaces.Box(0, 10, (), jnp.int32),
            "active_target_idx": spaces.Box(0, self.max_objs, (), jnp.int32)
        })

    def observation_space(self) -> spaces.Dict:
        return self._observation_space

    # --- 1. Event Oracle (The Logic Shift) ---
    def _detect_events(self, prev_state, curr_state):
        """
        Generates symbolic event flags for the reward function (like flagging a diver as collected).
        This enables the Reward Function to be 100% generic.
        """
        events = jnp.zeros(self.max_objs, dtype=jnp.int32)

        # Extract SeaquestState from AtariState wrapper
        prev_seaquest = prev_state.env_state
        curr_seaquest = curr_state.env_state

        # A. COLLECTION EVENTS (Divers: Indices 25-28)
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
        events = events.at[25:29].set(d_event.astype(jnp.int32))

        # B. DESTRUCTION EVENTS (Enemies: Sharks 1-12, Subs 13-24)
        # Logic: Score up + Enemy disappeared (and NOT a diver collection)
        # This prevents awarding "Destroy" when we actually "Collected"
        combat_score = jnp.logical_and(score_up, ~divers_increased)
        
        # Sharks - position arrays are (12, 3): [x, y, active_or_direction]
        s_prev_act = prev_seaquest.shark_positions[:, 2] != 0
        s_curr_act = curr_seaquest.shark_positions[:, 2] != 0
        s_gone = jnp.logical_and(s_prev_act, ~s_curr_act)
        events = events.at[1:13].set(jnp.logical_and(s_gone, combat_score).astype(jnp.int32))
        
        # Subs - position arrays are (12, 3): [x, y, active_or_direction]
        sub_prev_act = prev_seaquest.sub_positions[:, 2] != 0
        sub_curr_act = curr_seaquest.sub_positions[:, 2] != 0
        sub_gone = jnp.logical_and(sub_prev_act, ~sub_curr_act)
        events = events.at[13:25].set(jnp.logical_and(sub_gone, combat_score).astype(jnp.int32))

        # C. COLLISION / AVOID [FIXED]
        # Old: lives < prev_lives (Too slow, waits for animation)
        # New: death_counter just activated (0 -> >0)
        # Note: We verify lives didn't INCREASE (just in case of 10k point bonus)
        just_died = jnp.logical_and(prev_seaquest.death_counter == 0, curr_seaquest.death_counter > 0)
        
        # If we died, tag all active enemies as "Danger"
        if_hit_tag_sharks = jnp.logical_and(just_died, s_prev_act)
        if_hit_tag_subs = jnp.logical_and(just_died, sub_prev_act)
        
        # Check enemy missiles (indices 29-32)
        m_prev_act = prev_seaquest.enemy_missile_positions[:, 2] != 0
        if_hit_tag_missiles = jnp.logical_and(just_died, m_prev_act)

        # Apply tags
        events = events.at[1:13].max(if_hit_tag_sharks.astype(jnp.int32))
        events = events.at[13:25].max(if_hit_tag_subs.astype(jnp.int32))
        events = events.at[29:33].max(if_hit_tag_missiles.astype(jnp.int32))

        # D. MONITOR/SURFACE EVENT (Oxygen Refill)
        # Logic: Oxygen increased
        refilled = curr_seaquest.oxygen > prev_seaquest.oxygen
        
        # Ignore refill if we are in "Init State" (just_surfaced == -1) or if we are dead/dying (death_counter > 0)
        is_playing = jnp.logical_and(curr_seaquest.just_surfaced != -1, curr_seaquest.death_counter == 0)
        valid_refill = jnp.logical_and(refilled, is_playing)
        
        events = events.at[33].set(valid_refill.astype(jnp.int32))

        return events

    # --- 2. Normalization & Stacking ---
    def _normalize_feats(self, entity_array, label_id):
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
        
        labels = jnp.full((entity_array.shape[0],), label_id, dtype=jnp.int32)
        mask = active.astype(jnp.float32)
        return feats, mask, labels

    def _process_single_frame(self, obs, state):
        """Extracts features from ONE frame."""
        # Helper for scalars
        def sc(e): return jnp.array([[e.x, e.y, e.width, e.height, e.active]])

        # 1. Process Groups
        # Player (0)
        p_f, p_m, p_l = self._normalize_feats(sc(obs.player), const.LABEL_PLAYER)
        
        # Enemies -> DESTROY (or AVOID, depending on strategy. Let's use DESTROY as primary for enemies we can kill)
        # Actually, Sharks/Subs are usually DESTROY targets. 
        # But if we have no ammo, they become AVOID. 
        # For now, let's label them DESTROY. The policy can learn to avoid if it can't shoot.
        sh_f, sh_m, sh_l = self._normalize_feats(obs.sharks, const.LABEL_DESTROY)
        su_f, su_m, su_l = self._normalize_feats(obs.submarines, const.LABEL_DESTROY)
        
        # Divers -> COLLECT
        di_f, di_m, di_l = self._normalize_feats(obs.divers, const.LABEL_COLLECT)
        
        # Enemy Missiles -> AVOID (Cannot destroy them usually)
        em_f, em_m, em_l = self._normalize_feats(obs.enemy_missiles, const.LABEL_AVOID)
        
        # Surface Sub -> MONITOR STATISTIC
        ss_f, ss_m, ss_l = self._normalize_feats(sc(obs.surface_submarine), const.LABEL_MONITOR_STATISTIC)
        
        # Player Missile -> TOOL (or part of Destroy concept)
        # We can label it HIGH_LEVEL or just Generic.
        pm_f, pm_m, pm_l = self._normalize_feats(sc(obs.player_missile), const.LABEL_PLAYER_TOOL)

        # 2. Concat
        objects = jnp.concatenate([p_f, sh_f, su_f, di_f, em_f, ss_f, pm_f], axis=0)
        mask = jnp.concatenate([p_m, sh_m, su_m, di_m, em_m, ss_m, pm_m], axis=0)
        labels = jnp.concatenate([p_l, sh_l, su_l, di_l, em_l, ss_l, pm_l], axis=0)

        # 3. Global Features (Expert Dashboard)
        # We pass these to ALL sub-policies!
        oxy_norm = obs.oxygen_level / 64.0
        divers_held = obs.collected_divers / 6.0
        lives_norm = obs.lives / 3.0
        can_fire = (obs.player_missile.active == 0).astype(jnp.float32)
        
        # Calculate simple densities
        enemy_count = jnp.sum(sh_m) + jnp.sum(su_m)
        diver_count = jnp.sum(di_m)
        
        global_features = jnp.stack([
            oxy_norm, divers_held, lives_norm, can_fire,
            enemy_count / 24.0, diver_count / 4.0,
            obs.player.x / self.SCREEN_WIDTH, # Player Global X
            obs.player.y / self.SCREEN_HEIGHT, # Player Global Y
            0.0, 0.0 # Pad to 10 if needed
        ])

        return objects, mask, labels, global_features

    def _process_stack(self, obs_stack, events):
        """
        Flattens history, attaches events.
        """
        # VMAP over stack to get features [Stack, N, 4]
        # We need to pass a dummy state or just use the obs, but _process_single_frame only needs obs.
        # Wait, _process_single_frame calls obs.sharks...
        objects_stack, mask_stack, labels_stack, globals_stack = jax.vmap(
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

        # --- Compute Capabilities ---
        # Logic: If Player Missile (Idx 34) is present (Mask=1) -> Can't Fire.
        # Check specific index for efficiency since indices are fixed in Seaquest
        missile_active = latest_mask[self.idx_player_missile] > 0.5
        can_fire = jnp.where(missile_active, 0.0, 1.0)
        
        caps = jnp.zeros(const.N_CAPABILITIES)
        caps = caps.at[const.CAP_CAN_FIRE].set(can_fire)

        # Events should always be a JAX array (zeros on reset, detected events on step)
        # No need for None check - always a valid JAX array

        return ConceptObs(
            global_features=latest_globals,
            capabilities=caps, # <--- Added
            objects=flat_objects,
            object_mask=latest_mask,
            object_labels=latest_labels,
            object_events=events, # <--- The Magic Link
            active_concept=jnp.array(0, dtype=jnp.int32),
            active_target_idx=jnp.array(0, dtype=jnp.int32)
        )

    # --- 3. Main Loop Overrides ---
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        obs_stack, state = self._env.reset(key)
        # No events on reset - pass zero array
        zero_events = jnp.zeros(self.max_objs, dtype=jnp.int32)
        return self._process_stack(obs_stack, zero_events), state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # 1. Capture Previous State (for Event Detection)
        prev_state = state
        
        # 2. Step
        obs_stack, next_state, reward, done, info = self._env.step(state, action)
        
        # 3. Detect Events (The Oracle)
        events = self._detect_events(prev_state, next_state)
        
        # 4. Process
        concept_obs = self._process_stack(obs_stack, events)
        
        return concept_obs, next_state, reward, done, info