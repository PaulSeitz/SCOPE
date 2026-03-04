import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pygame
import wandb
from omegaconf import OmegaConf
from safetensors.flax import load_file
from flax.traverse_util import unflatten_dict

from jaxatari import make as make_atari_env
from jaxatari.wrappers import AtariWrapper
from jaxatari.games.jax_seaquest import SeaquestRenderer

from concept_based_learning.egocentric_transformation import get_ego_obs
from concept_based_learning.environments.datatypes import ConceptObservation
from concept_based_learning.environments.seaquest.seaquest_label import (
    SeaquestSemanticWrapper,
    seaquest_action_mapper,
)
from concept_based_learning.environments.seaquest.seaquest_policy import (
    get_heuristic_target,
    NO_TARGET_IDX,
)
from concept_based_learning.train_agent import QNetwork
from concept_based_learning.video_recorder import (
    video_callback,
    wait_for_video_finish,
    draw_target_outline,
)


def _project_root() -> Path:
    """Return the repository root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


def load_base_config():
    """
    Load the global config and the default Seaquest experiment config,
    then merge them into a single flat dict similar to `single_run` in main.py.
    """
    root = _project_root()

    # Global config (entity/project/WANDB/video settings, etc.)
    global_cfg_path = root / "configs" / "config.yaml"
    global_cfg = OmegaConf.load(global_cfg_path)

    # Determine default experiment from Hydra-style defaults list
    # Hydra-style defaults is a list of DictConfig entries.
    # They are not plain dicts, so we must not restrict the type check.
    defaults = global_cfg.get("defaults", [])
    exp_name = None
    for entry in defaults:
        # `entry` is typically a DictConfig; use .get instead of isinstance(..., dict).
        if hasattr(entry, "get"):
            maybe_exp = entry.get("experiment", None)
            if maybe_exp is not None:
                exp_name = maybe_exp
                break
    if exp_name is None:
        raise ValueError("Could not find an 'experiment' entry in configs/config.yaml defaults.")

    exp_cfg_path = root / "configs" / "experiment" / f"{exp_name}.yaml"
    exp_cfg = OmegaConf.load(exp_cfg_path)

    global_dict = OmegaConf.to_container(global_cfg, resolve=True)
    exp_dict = OmegaConf.to_container(exp_cfg, resolve=True)

    # Flatten like main.single_run: base + experiment + experiment.alg
    base_config = {**global_dict, **exp_dict, **exp_dict.get("alg", {})}
    return base_config


def load_params_from_file(filename):
    """
    Load a PyTree of parameters from a .safetensors file.

    This expects that the file was written with `safetensors.flax.save_file`
    on a parameter pytree that matches the skill Q-network params used in training
    (typically something like `train_states.params` from the trainer).
    """
    print(f"Loading weights from: {filename}")
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


def create_eval_env(config):
    """
    Create the Seaquest evaluation environment + renderer using the same
    wrappers as the new training pipeline (AtariWrapper + SeaquestSemanticWrapper).
    """
    game_mods = config.get("GAME_MODS", None)
    sticky_actions = config.get("STICKY_ACTIONS", False)
    episodic_life = config.get("EPISODIC_LIFE", False)

    if game_mods:
        env = make_atari_env("seaquest", mods=set(game_mods))
    else:
        env = make_atari_env("seaquest")

    # Evaluation uses a non-sticky, non-episodic-life wrapper for clean rollouts,
    # mirroring `outer_make_train(train=False)` in main.py.
    env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
    env = SeaquestSemanticWrapper(env)

    renderer = SeaquestRenderer()
    return env, renderer


def label_to_skill_idx(label, num_skills: int):
    """
    Map semantic labels to skill indices, matching the convention in train_agent.make_train:
    0 = NOOP, 1 = DESTROY, 2 = COLLECT, 3 = AVOID  → skills 0..num_skills-1 (no NOOP skill).
    """
    return jnp.clip(label - 1, 0, num_skills - 1)


def run_eval_rollout(config, params_tree=None, max_frames=None):
    """
    Run a single evaluation rollout with the hierarchical PQN-style agent:
    - Fixed heuristic meta-policy to choose a target object.
    - Shared skill Q-network (one set of weights per skill).
    - Ego-centric action space mapped back to Atari actions.

    Returns the same tensors that `video_recorder.video_callback` expects.
    """
    num_skills = config["NUM_SKILLS"]
    ego_action_dim = config["EGO_ACTION_DIM"]
    hidden_size = config["HIDDEN_SIZE"]
    num_layers = config["NUM_LAYERS"]
    seed = config["SEED"]
    max_frames = max_frames or config.get("VIDEO_MAX_FRAMES", 2000)

    env, renderer = create_eval_env(config)

    # Build the shared skill Q-network (one set of params per skill)
    skill_network = QNetwork(
        action_dim=ego_action_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    rng = jax.random.PRNGKey(seed)
    rng, rng_reset = jax.random.split(rng)
    obs, state = env.reset(rng_reset)

    # Infer ego feature dimension from a dummy egocentric observation
    # using target index 0 as a placeholder.
    dummy_ego = get_ego_obs(
        obs.raw_observation,
        obs.objects,
        jnp.array(0, dtype=jnp.int32),
    )
    ego_feat_dim = int(dummy_ego.shape[-1])

    # Initialize parameters if none were provided.
    if params_tree is None:
        init_x = jnp.zeros(ego_feat_dim)
        network_vars = skill_network.init(jax.random.PRNGKey(seed + 1), init_x, train=False)
        base_params = network_vars["params"]

        # Replicate the same initialization across all skills.
        params_tree = jax.tree_util.tree_map(
            lambda p: jnp.stack([p] * num_skills, axis=0),
            base_params,
        )

    # Rollout storage
    states_list = []
    active_list = []
    combined_qs_list = []
    dones_list = []
    distances_list = []
    angles_list = []
    target_obj_idxs_list = []
    obj_x_list = []
    obj_y_list = []
    obj_w_list = []
    obj_h_list = []
    obj_active_list = []

    # Persistent target index for the heuristic meta-policy
    prev_target_idx = NO_TARGET_IDX

    for _ in range(max_frames):
        rng, step_key = jax.random.split(rng)

        # Heuristic meta-policy selects a target object
        target_obj_idx = get_heuristic_target(
            obs.raw_observation,
            obs.objects,
            prev_target_idx,
            step_key,
        )
        target_obj_idx = int(jnp.ravel(target_obj_idx)[0])
        prev_target_idx = target_obj_idx

        if target_obj_idx == NO_TARGET_IDX:
            # No valid target on screen → NOOP, dummy Q values.
            global_action = 0
            active_skill_idx = 0
            combined_q = np.zeros(num_skills, dtype=np.float32)
            distance = np.nan
            angle = np.nan
        else:
            # Choose skill from the semantic label of the current target object
            raw_label = obs.objects.semantic_label[-1, target_obj_idx]
            active_skill_idx = int(label_to_skill_idx(raw_label, num_skills))

            # Egocentric observation relative to the target
            ego_obs = get_ego_obs(
                obs.raw_observation,
                obs.objects,
                jnp.array(target_obj_idx, dtype=jnp.int32),
            )
            ego_obs_batch = ego_obs[jnp.newaxis, ...]

            # Extract the most recent frame's distance and angle from the ego features.
            # Each frame contributes [distance/264, angle/pi, sin(ori), cos(ori)];
            # the last 4 entries correspond to the latest frame.
            last_frame_ego = ego_obs[-4:]
            distance = float(last_frame_ego[0] * 264.0)
            angle = float(last_frame_ego[1] * jnp.pi)

            # Evaluate all skills' Q-values on the current ego observation.
            def apply_one(params_for_skill):
                return skill_network.apply(
                    {"params": params_for_skill},
                    ego_obs_batch,
                    train=False,
                )

            q_vals = jax.vmap(apply_one)(params_tree)  # [num_skills, 1, ego_action_dim]
            q_vals = jnp.squeeze(q_vals, axis=1)      # [num_skills, ego_action_dim]

            combined_q = np.array(jnp.max(q_vals, axis=-1))
            chosen_ego = int(jnp.argmax(q_vals[active_skill_idx]))
            global_action = int(seaquest_action_mapper(chosen_ego, obs.raw_observation))

        # Step the environment
        obs, state, _reward, done, _info = env.step(state, global_action)

        # Record state & debug info
        states_list.append(state)
        active_list.append(active_skill_idx)
        combined_qs_list.append(combined_q.astype(np.float32))
        dones_list.append(float(done))
        distances_list.append(distance)
        angles_list.append(angle)
        target_obj_idxs_list.append(target_obj_idx)

        # Object bounding boxes after the step
        obj_x_list.append(np.array(obs.objects.x[-1]))
        obj_y_list.append(np.array(obs.objects.y[-1]))
        obj_w_list.append(np.array(obs.objects.width[-1]))
        obj_h_list.append(np.array(obs.objects.height[-1]))
        obj_active_list.append(np.array(obs.objects.active[-1]))

        if done:
            break

    if not states_list:
        raise RuntimeError("No frames collected during evaluation rollout.")

    def stack_trees(*trees):
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)

    stacked_states = stack_trees(*states_list)
    active_agents = jnp.array(active_list, dtype=jnp.int32)
    combined_qs = jnp.stack([jnp.array(q) for q in combined_qs_list])
    dones = jnp.array(dones_list, dtype=jnp.float32)
    distances = np.array(distances_list, dtype=np.float32)
    angles = np.array(angles_list, dtype=np.float32)
    target_obj_idxs_stacked = np.array(target_obj_idxs_list, dtype=np.int32)
    objects_bbox = (
        np.stack(obj_x_list),
        np.stack(obj_y_list),
        np.stack(obj_w_list),
        np.stack(obj_h_list),
        np.stack(obj_active_list),
    )

    return (
        stacked_states,
        active_agents,
        combined_qs,
        dones,
        target_obj_idxs_stacked,
        objects_bbox,
        distances,
        angles,
        renderer,
    )


def run_live_visualization(config, params_tree=None, max_frames=None):
    """
    Run a live, interactive visualization loop using pygame.
    The agent acts in the environment step-by-step and the current frame is
    rendered to a window, with the skill input (distance, angle) shown at the bottom.
    """
    num_skills = config["NUM_SKILLS"]
    ego_action_dim = config["EGO_ACTION_DIM"]
    hidden_size = config["HIDDEN_SIZE"]
    num_layers = config["NUM_LAYERS"]
    seed = config["SEED"]
    max_frames = max_frames or config.get("VIDEO_MAX_FRAMES", 2000)

    env, renderer = create_eval_env(config)

    skill_network = QNetwork(
        action_dim=ego_action_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    rng = jax.random.PRNGKey(seed)
    rng, rng_reset = jax.random.split(rng)
    obs, state = env.reset(rng_reset)

    dummy_ego = get_ego_obs(
        obs.raw_observation,
        obs.objects,
        jnp.array(0, dtype=jnp.int32),
    )
    ego_feat_dim = int(dummy_ego.shape[-1])

    if params_tree is None:
        init_x = jnp.zeros(ego_feat_dim)
        network_vars = skill_network.init(jax.random.PRNGKey(seed + 1), init_x, train=False)
        base_params = network_vars["params"]
        params_tree = jax.tree_util.tree_map(
            lambda p: jnp.stack([p] * num_skills, axis=0),
            base_params,
        )

    # Pygame setup
    pygame.init()
    pygame.display.set_caption("Hierarchical PQN Live Visualization - Seaquest")
    font = pygame.font.Font(None, 20)

    # Get one rendered frame to determine window size
    base_state = state.atari_state.env_state
    sample_frame = renderer.render(base_state)
    sample_frame = np.array(jax.device_get(sample_frame), dtype=np.uint8)
    if sample_frame.ndim == 2:
        sample_frame = np.stack([sample_frame] * 3, axis=-1)
    H, W, _ = sample_frame.shape
    scale = 2
    window_size = (W * scale, H * scale)
    screen = pygame.display.set_mode(window_size)

    clock = pygame.time.Clock()
    running = True
    prev_target_idx = NO_TARGET_IDX
    frame_count = 0

    while running and frame_count < max_frames:
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

        rng, step_key = jax.random.split(rng)

        # Heuristic meta-policy selects a target object
        target_obj_idx = get_heuristic_target(
            obs.raw_observation,
            obs.objects,
            prev_target_idx,
            step_key,
        )
        target_obj_idx = int(jnp.ravel(target_obj_idx)[0])
        prev_target_idx = target_obj_idx

        if target_obj_idx == NO_TARGET_IDX:
            global_action = 0
            active_skill_idx = 0
            distance = np.nan
            angle = np.nan
        else:
            raw_label = obs.objects.semantic_label[-1, target_obj_idx]
            active_skill_idx = int(label_to_skill_idx(raw_label, num_skills))

            ego_obs = get_ego_obs(
                obs.raw_observation,
                obs.objects,
                jnp.array(target_obj_idx, dtype=jnp.int32),
            )
            ego_obs_batch = ego_obs[jnp.newaxis, ...]

            last_frame_ego = ego_obs[-4:]
            distance = float(last_frame_ego[0] * 264.0)
            angle = float(last_frame_ego[1] * jnp.pi)

            def apply_one(params_for_skill):
                return skill_network.apply(
                    {"params": params_for_skill},
                    ego_obs_batch,
                    train=False,
                )

            q_vals = jax.vmap(apply_one)(params_tree)
            q_vals = jnp.squeeze(q_vals, axis=1)

            chosen_ego = int(jnp.argmax(q_vals[active_skill_idx]))
            global_action = int(seaquest_action_mapper(chosen_ego, obs.raw_observation))

        # Step env
        obs, state, _reward, done, _info = env.step(state, global_action)

        # Render current frame
        base_state = state.atari_state.env_state
        frame = renderer.render(base_state)
        frame = np.array(jax.device_get(frame), dtype=np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        # Add a red border around the selected target object (same style as videos).
        if target_obj_idx != NO_TARGET_IDX:
            ox = np.array(obs.objects.x[-1])
            oy = np.array(obs.objects.y[-1])
            ow = np.array(obs.objects.width[-1])
            oh = np.array(obs.objects.height[-1])
            oactive = np.array(obs.objects.active[-1])

            frame_chw = np.transpose(frame, (2, 0, 1))
            frame_chw = draw_target_outline(
                frame_chw,
                target_obj_idx,
                ox,
                oy,
                ow,
                oh,
                oactive,
            )
            frame = np.transpose(frame_chw, (1, 2, 0))

        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, window_size)
        screen.blit(surf, (0, 0))

        # Draw distance/angle text at bottom-center (if a valid target exists)
        if not np.isnan(distance):
            text = f"skill: d={distance:.1f}, a={angle:.2f}"
            text_surface = font.render(text, True, (255, 255, 255))
            tx = (window_size[0] - text_surface.get_width()) // 2
            ty = window_size[1] - text_surface.get_height() - 10
            screen.blit(text_surface, (tx, ty))

        pygame.display.flip()
        clock.tick(30)
        frame_count += 1

        if done:
            rng, rng_reset = jax.random.split(rng)
            obs, state = env.reset(rng_reset)
            prev_target_idx = NO_TARGET_IDX

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical PQN visualizer for Seaquest using the new agent structure."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional path to a .safetensors file containing skill Q-network parameters "
             "(expects the same tree structure as training). If omitted, uses random init.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to render in the video (defaults to VIDEO_MAX_FRAMES from config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed (defaults to SEED from config).",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="disabled",
        help="Override WANDB_MODE from config. Default is 'disabled' for local-only visualization.",
    )
    parser.add_argument(
        "--modpack",
        type=str,
        choices=["base", "collect", "destroy", "avoid"],
        default="destroy",
        help="Which Seaquest modpack to use: one of 'base', 'collect', 'destroy', or 'avoid'.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run a live interactive visualization instead of logging a video.",
    )
    args = parser.parse_args()

    base_config = load_base_config()
    if args.seed is not None:
        base_config["SEED"] = args.seed

    # Override WANDB mode for this run (default: disabled for local-only use).
    if args.wandb_mode is not None:
        base_config["WANDB_MODE"] = args.wandb_mode

    # Select which modpack to run (base env vs one of the curriculum packs).
    if args.modpack != "base":
        curriculum_mods = base_config.get("CURRICULUM_MODS", {})
        game_mods = curriculum_mods.get(args.modpack)
        if game_mods is None:
            raise ValueError(
                f"Requested modpack '{args.modpack}' not found in CURRICULUM_MODS "
                f"(available keys: {list(curriculum_mods.keys())})."
            )
        base_config["GAME_MODS"] = game_mods

    # Optional parameter loading
    params_tree = None
    if args.model is not None:
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model not found: {args.model}")
        params_tree = load_params_from_file(args.model)

    # If requested, run live interactive visualization and exit.
    if args.live:
        run_live_visualization(base_config, params_tree=params_tree, max_frames=args.max_frames)
        return

    # Initialize wandb so that `video_recorder` can log the video (offline by default).
    if base_config.get("WANDB_MODE", "disabled") != "disabled":
        wandb.init(
            entity=base_config.get("ENTITY", None),
            project=base_config.get("PROJECT", "concept_based_learning"),
            tags=[
                base_config.get("ALG_NAME", "hierarchical_pqn").upper(),
                base_config.get("ENV_NAME", "Seaquest").upper(),
                f"visualizer",
            ],
            name=f'{base_config.get("ALG_NAME", "hierarchical_pqn")}_visualizer',
            config=base_config,
            mode=base_config["WANDB_MODE"],
        )

    print("Running evaluation rollout for visualization...")
    (
        states,
        active_agents,
        combined_qs,
        dones,
        target_obj_idxs,
        objects_bbox,
        distances,
        angles,
        renderer,
    ) = run_eval_rollout(base_config, params_tree=params_tree, max_frames=args.max_frames)

    # Use the existing video logging utility to create and log the video.
    step = 0
    video_callback(
        states,
        active_agents,
        combined_qs,
        dones,
        step,
        renderer,
        target_obj_idxs=target_obj_idxs,
        objects_bbox=objects_bbox,
        distances=distances,
        angles=angles,
    )
    wait_for_video_finish()

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()