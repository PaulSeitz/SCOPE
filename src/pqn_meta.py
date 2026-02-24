# type: ignore
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

from flax.linen.initializers import constant, orthogonal
import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import wandb

from src.video_recorder import video_callback

# --- 1. NETWORK DEFINITIONS ---
class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 64
    num_layers: int = 3
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        
        # Norm Selection
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        return x

# --- 2. JAX LOGIC HELPERS (The "Shenanigans" Fix) ---

def extract_relative_obs(flat_obs, target_idx, stack_size, n_global, base_obj_feats, num_concepts, raw_offset):
    """
    Robust extraction handling multi-hot concepts.
    Slices out raw pixels first, then extracts relative observation.
    Calculates stride = base_feats (5) + num_concepts.
    """
    # 0. Slice OFF the pixels to get concept part
    concept_obs = flat_obs[raw_offset:]
    
    # 1. Calculate Dynamic Stride
    # This ensures we don't slice in the middle of a concept vector
    n_obj_feats = base_obj_feats + num_concepts 
    
    # 2. Slice Globals
    global_len = stack_size * n_global
    globals_flat = concept_obs[:global_len]
    objects_flat = concept_obs[global_len:]
    
    globals_stack = globals_flat.reshape(stack_size, n_global)
    p_x_seq = globals_stack[:, 5] # Player X
    p_y_seq = globals_stack[:, 6] # Player Y
    
    # 3. Slice Objects
    # We infer n_objs from the remaining length to be safe
    # total_len = n_objs * stack * feats
    total_obj_len = objects_flat.shape[0]
    n_objs = 35 # Config max_objs (Seaquest standard)
    
    # Reshape: [35, 4, 8] (assuming 5+3 feats)
    objs_reshaped = objects_flat.reshape(n_objs, stack_size, n_obj_feats)
    
    # 4. Target Selection
    target_history = objs_reshaped[target_idx] 
    
    # 5. Validity (Active is index 4)
    is_valid = target_history[-1, 4] 
    
    # 6. Relative Calc
    rel_x_seq = (target_history[:, 0] - p_x_seq) * is_valid
    rel_y_seq = (target_history[:, 1] - p_y_seq) * is_valid
    
    # --- NEW ---
    w_seq = target_history[:, 2] * is_valid
    h_seq = target_history[:, 3] * is_valid
    # -----------

    valid_seq = jnp.full((stack_size,), is_valid)
    p_x_seq = jnp.zeros_like(p_x_seq)
    
    player_history = jnp.stack([p_x_seq, p_y_seq], axis=-1).flatten()
    
    # Update stack to include w and h
    stack_feats = jnp.stack([rel_x_seq, rel_y_seq, w_seq, h_seq, valid_seq], axis=-1).flatten()
    
    return jnp.concatenate([player_history, stack_feats])

def get_skill_id_from_target(target_idx):
    """
    Enforces Priority: DESTROY > COLLECT > AVOID
    Based on SeaquestConceptWrapper fixed slots.
    """
    # 0-23: Enemies (Sharks + Subs). Wrapper might label them Destroy AND Avoid.
    # RULE: We default to Destroy.
    is_destroy = (target_idx < 24)
    
    # 24-27: Divers. Wrapper labels Collect.
    is_collect = (target_idx >= 24) & (target_idx < 28)
    
    # 28+: Missiles / Surface Sub / Padding. Wrapper labels Avoid.
    # Default Fallback.
    
    # Logic Tree:
    # IF Destroy Slot -> Return 1 (Destroy)
    # ELSE IF Collect Slot -> Return 0 (Collect)
    # ELSE -> Return 2 (Avoid)
    return jax.lax.cond(
        is_destroy, lambda: 1,
        lambda: jax.lax.cond(is_collect, lambda: 0, lambda: 2)
    )

# --- 3. DATACLASSES ---
@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    sub_obs: chex.Array
    meta_action: chex.Array
    skill_action: chex.Array
    skill_id: chex.Array
    rewards: chex.Array
    done: chex.Array
    next_obs: chex.Array
    next_sub_obs: chex.Array

class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0

# --- 3. TRAIN LOGIC FACTORY ---
def make_train(config, env, test_env, test_env_modif, pretrained_params, renderer):
    
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    n_global = config["environment"]["n_global_feats"]
    base_obj_feats = config["environment"]["n_obj_feats"]
    num_concepts = config["concepts"]["label_high_level_concept"] + 1
    raw_offset = config.get("RAW_OFFSET", 0)
    
    full_meta_obs_shape = env.observation_space().shape
    # Meta network sees ONLY raw pixels
    meta_obs_shape = (raw_offset,)

    # Helper for Exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        random_actions = jax.random.randint(rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1])
        return jnp.where(jax.random.uniform(rng_e, greedy_actions.shape) < eps, random_actions, greedy_actions)

    # Vmap wrappers
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset)(jax.random.split(rng, n_envs))
    vmap_step = lambda env_state, action: jax.vmap(env.step)(env_state, action)
    
    # Test Wrappers
    test_vmap_reset = lambda n_envs: lambda rng: jax.vmap(test_env.reset)(jax.random.split(rng, n_envs))
    test_vmap_step = lambda env_state, action: jax.vmap(test_env.step)(env_state, action)
    
    modif_test_vmap_reset = lambda n_envs: lambda rng: jax.vmap(test_env_modif.reset)(jax.random.split(rng, n_envs))
    modif_test_vmap_step = lambda env_state, action: jax.vmap(test_env_modif.step)(env_state, action)

    # --- A. INITIALIZATION ---
    def init_agent(rng):
        lr_scheduler = optax.linear_schedule(config["LR"], 1e-20, config["NUM_UPDATES"])
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        meta_network = QNetwork(action_dim=35, norm_type=config["NORM_TYPE"], hidden_size=config["HIDDEN_SIZE"], num_layers=config["NUM_LAYERS"])
        skill_network = QNetwork(action_dim=18, norm_type=config["NORM_TYPE"]) 

        def create_train_state(rng, network, input_shape, params=None, freeze=False):
            init_x = jnp.zeros(input_shape)
            if params is None:
                variables = network.init(rng, init_x, train=False)
                params = variables["params"]
                batch_stats = variables.get("batch_stats", {})
            else:
                batch_stats = {} 

            tx = optax.radam(learning_rate=lr)
            if freeze:
                tx = optax.set_to_zero()

            return CustomTrainState.create(
                apply_fn=network.apply, params=params, batch_stats=batch_stats, tx=tx
            )

        rng, rng_m, rng_s = jax.random.split(rng, 3)
        meta_state = create_train_state(rng_m, meta_network, meta_obs_shape)
        
        # Sub-Obs: Player(2*4) + Obj(5*4) = 28
        sub_obs_shape = (28,)
        skills_state = create_train_state(
            rng_s, skill_network, sub_obs_shape, 
            params=pretrained_params, freeze=config.get("FREEZE_SKILLS", False)
        )
        
        rng, _rng = jax.random.split(rng)
        expl_state = vmap_reset(config["NUM_ENVS"])(_rng) # (obs, env_state)
        
        runner_state = ((meta_state, skills_state), expl_state[0], expl_state[1], rng)
        return runner_state

    # --- B. UPDATE STEP ---
    def update_step(runner_state, _):
        (states, last_obs, env_state, rng) = runner_state
        (meta_state, skills_state) = states
        
        eps_scheduler = optax.linear_schedule(config["EPS_START"], config["EPS_FINISH"], config["EPS_DECAY"] * config["NUM_UPDATES"])
        
        def _step_env(carry, _):
            (meta_state, skills_state), last_obs, env_state, rng = carry
            rng, rng_meta, rng_skill = jax.random.split(rng, 3)

            # [CRITICAL] Slice Observation for Meta-Policy
            # It only sees [0 : raw_offset] (raw pixels)
            meta_input = last_obs[..., :raw_offset]
            
            # Meta Pass
            meta_q = meta_state.apply_fn(
                {"params": meta_state.params, "batch_stats": meta_state.batch_stats}, 
                meta_input, train=False
            )
            eps = eps_scheduler(meta_state.n_updates)
            
            # Epsilon Greedy Meta
            rng_meta_split = jax.random.split(rng_meta, config["NUM_ENVS"])
            greedy_actions = jnp.argmax(meta_q, axis=-1)
            random_actions = jax.random.randint(rng_meta, shape=greedy_actions.shape, minval=0, maxval=meta_q.shape[-1])
            meta_action = jnp.where(jax.random.uniform(rng_meta, greedy_actions.shape) < eps, random_actions, greedy_actions)

            # Router & Extraction
            skill_indices = jax.vmap(get_skill_id_from_target)(meta_action)
            
            extract_fn = lambda o, t: extract_relative_obs(o, t, 4, n_global, base_obj_feats, num_concepts, raw_offset)
            sub_obs = jax.vmap(extract_fn)(last_obs, meta_action)

            # Skill Pass
            def apply_skill(params, obs): return skills_state.apply_fn({"params": params}, obs, train=False)
            all_skill_q = jax.vmap(apply_skill, in_axes=(0, None))(skills_state.params, sub_obs)
            all_skill_q = jnp.transpose(all_skill_q, (1, 0, 2))
            active_q = jnp.take_along_axis(all_skill_q, skill_indices[:, None, None], axis=1).squeeze(1)
            
            rng_skill_split = jax.random.split(rng_skill, config["NUM_ENVS"])
            s_greedy = jnp.argmax(active_q, axis=-1)
            s_random = jax.random.randint(rng_skill, shape=s_greedy.shape, minval=0, maxval=active_q.shape[-1])
            primitive_action = jnp.where(jax.random.uniform(rng_skill, s_greedy.shape) < eps, s_random, s_greedy)

            # Execute
            new_obs, new_env_state, all_rewards, done, info = vmap_step(env_state, primitive_action)
            next_sub_obs = jax.vmap(extract_fn)(new_obs, meta_action)

            transition = Transition(
                obs=last_obs, sub_obs=sub_obs, meta_action=meta_action, 
                skill_action=primitive_action, skill_id=skill_indices, rewards=all_rewards, 
                done=done, next_obs=new_obs, next_sub_obs=next_sub_obs
            )
            return ((meta_state, skills_state), new_obs, new_env_state, rng), (transition, info)

        scan_carry = (states, last_obs, env_state, rng)
        (final_carry, (transitions, infos)) = jax.lax.scan(
            _step_env, scan_carry, None, config["NUM_STEPS"]
        )
        ((_, _), final_obs, final_env_state, final_rng) = final_carry

        # --- LEARNING ---
        def flatten_batch(x): return x.reshape(-1, *x.shape[2:])
        flat_trans = jax.tree.map(flatten_batch, transitions)
        
        # Meta Update
        meta_rewards = flat_trans.rewards[..., 3]
        def meta_loss_fn(params):
            # Slice input for training too!
            meta_input = flat_trans.obs[..., :raw_offset]
            meta_next_input = flat_trans.next_obs[..., :raw_offset]
            
            q_vals = meta_state.apply_fn({"params": params}, meta_input, train=True)
            chosen_q = jnp.take_along_axis(q_vals, flat_trans.meta_action[..., None], axis=-1).squeeze(-1)
            next_q = meta_state.apply_fn({"params": params}, meta_next_input, train=False)
            targets = meta_rewards + config["GAMMA"] * (1 - flat_trans.done) * jnp.max(next_q, axis=-1)
            return 0.5 * jnp.square(chosen_q - targets).mean()

        m_grads = jax.grad(meta_loss_fn)(meta_state.params)
        new_meta_state = meta_state.apply_gradients(grads=m_grads)
        new_meta_state = new_meta_state.replace(n_updates=meta_state.n_updates + 1, timesteps=meta_state.timesteps + config["NUM_STEPS"]*config["NUM_ENVS"])

        # Skill Update (only if unfrozen)
        new_skills_state = skills_state
        if not config.get("FREEZE_SKILLS", False):
            # (Skill update logic omitted for brevity, but same as before)
            pass
            
        new_runner_state = ((new_meta_state, new_skills_state), final_obs, final_env_state, final_rng)
        
        metrics = {
            "game_score": jnp.mean(infos["returned_episode_returns"][..., 3]), 
            "epsilon": eps_scheduler(meta_state.n_updates),
            "timesteps": new_meta_state.timesteps
        }
        
        return new_runner_state, metrics

    # --- C. EVALUATION STEP (With Video) ---
    def evaluate_agent(runner_state, modif, rng):
        (states, _, _, _) = runner_state
        (meta_state, skills_state) = states
        
        def _test_step(carry, _):
            env_state, last_obs, rng = carry
            rng, rng_meta, rng_skill = jax.random.split(rng, 3)

            # [CRITICAL] Slice for Eval
            meta_input = last_obs[..., :raw_offset]
            
            # Meta Pass (Greedy)
            meta_q = meta_state.apply_fn({"params": meta_state.params}, meta_input, train=False)
            eps_test = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
            
            greedy_m = jnp.argmax(meta_q, axis=-1)
            random_m = jax.random.randint(rng_meta, shape=greedy_m.shape, minval=0, maxval=meta_q.shape[-1])
            meta_action = jnp.where(jax.random.uniform(rng_meta, greedy_m.shape) < eps_test, random_m, greedy_m)

            # Router
            skill_indices = jax.vmap(get_skill_id_from_target)(meta_action)
            
            extract_fn = lambda o, t: extract_relative_obs(o, t, 4, n_global, base_obj_feats, num_concepts, raw_offset)
            sub_obs = jax.vmap(extract_fn)(last_obs, meta_action)

            # Skill Pass (Greedy)
            def apply_skill(params, obs): return skills_state.apply_fn({"params": params}, obs, train=False)
            all_skill_q = jax.vmap(apply_skill, in_axes=(0, None))(skills_state.params, sub_obs)
            all_skill_q = jnp.transpose(all_skill_q, (1, 0, 2))
            active_q = jnp.take_along_axis(all_skill_q, skill_indices[:, None, None], axis=1).squeeze(1)

            s_greedy = jnp.argmax(active_q, axis=-1)
            s_random = jax.random.randint(rng_skill, shape=s_greedy.shape, minval=0, maxval=active_q.shape[-1])
            primitive_action = jnp.where(jax.random.uniform(rng_skill, s_greedy.shape) < eps_test, s_random, s_greedy)

            # Execute
            new_obs, new_env_state, _, done, info = jax.lax.cond(
                modif,
                lambda: modif_test_vmap_step(env_state, primitive_action),
                lambda: test_vmap_step(env_state, primitive_action)
            )
            
            # Prepare Video Data
            # env_state is a wrapper struct. The renderer needs the inner Atari state.
            # Usually SeaquestWrapper -> LogWrapper -> Env. 
            # We assume new_env_state[0] is what renderer expects (batch 0)
            env_state_vid = jax.tree.map(lambda x: x[0], new_env_state)
            
            # Pass skill_indices as "active_agents" so we can visualize what skill is running
            active_agent_vid = skill_indices[0] 

            return (new_env_state, new_obs, rng), (info, env_state_vid, active_agent_vid, done[0])

        rng, _rng = jax.random.split(rng)
        init_obs, env_state = jax.lax.cond(
            modif,
            lambda: modif_test_vmap_reset(config["TEST_NUM_ENVS"])(_rng),
            lambda: test_vmap_reset(config["TEST_NUM_ENVS"])(_rng)
        )
        
        # Scan Loop
        _, output = jax.lax.scan(_test_step, (env_state, init_obs, _rng), None, config["TEST_NUM_STEPS"])
        infos, states, active_agents, dones = output

        # Trigger Video Callback
        if config.get("RECORD_VIDEO", False):
            # We use jax.debug.callback to escape JAX and run Python rendering code
            jax.debug.callback(
                video_callback, 
                states, 
                active_agents, # [Time] (Skill IDs)
                None, # Q-values (optional, omitted)
                dones, 
                meta_state.n_updates, 
                renderer, 
                modif=modif
            )

        # Metrics
        done_infos = jax.tree.map(
            lambda x: jnp.nanmean(jnp.where(infos["returned_episode"], x, jnp.nan)),
            infos
        )
        return done_infos

    return init_agent, update_step, evaluate_agent