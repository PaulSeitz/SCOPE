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

from concept_based_learning.observations.action_space import fire_ego_action_indices
from concept_based_learning.observations.egocentric import (
    MAX_DIST,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    ego_obs_channels_per_frame,
    _get_player,
    _center_y,
    _uses_bottom_edge_y,
    get_ego_obs,
    set_ego_obs_format,
    set_ego_size_features,
    set_ego_velocity_features,
    set_ego_distance_clip_max_pixels,
    distance_pixels_from_feature,
    using_legacy_ego_obs,
)
from concept_based_learning.environments.datatypes import ConceptObservation
from concept_based_learning.environments.env_manager import EnvManager
from concept_based_learning.environments.choppercommand.choppercommand_policy import (
    debug_heuristic_snapshot,
)
from concept_based_learning.networks.meta_factory import create_meta_network
from concept_based_learning.environments.meta_target_mask import build_meta_target_mask
from concept_based_learning.networks.skill_network import QNetwork
from concept_based_learning.video_recorder import (
    video_callback,
    wait_for_video_finish,
    draw_target_outline,
)


def _project_root() -> Path:
    """Return the repository root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


def load_base_config(experiment: str | None = None):
    """
    Load the global config and an experiment config,
    then merge them into a single flat dict similar to `single_run` in main.py.
    """
    root = _project_root()

    # Global config (entity/project/WANDB/video settings, etc.)
    global_cfg_path = root / "configs" / "config.yaml"
    global_cfg = OmegaConf.load(global_cfg_path)

    exp_name = experiment
    if exp_name is None:
        # Determine default experiment from Hydra-style defaults list
        defaults = global_cfg.get("defaults", [])
        for entry in defaults:
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


def resolve_config_path(path):
    """
    Resolve a config/CLI path for loading or saving artifacts.

    Hydra changes the process working directory to each run's output folder
    (e.g. ``outputs/YYYY-MM-DD/HH-MM-SS/``). Relative paths like ``./outputs/models/...``
    must therefore be interpreted relative to the directory where the user
    launched the job, not the Hydra run directory.

    - Absolute paths are returned normalized unchanged.
    - Relative paths are joined with Hydra's original cwd when available,
      otherwise ``os.getcwd()``.
    """
    if path is None:
        return None
    if not isinstance(path, str):
        path = str(path)
    path = path.strip()
    if not path:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    try:
        from hydra.utils import get_original_cwd

        base = get_original_cwd()
    except Exception:
        base = os.getcwd()
    return os.path.normpath(os.path.join(base, path))


def load_params_from_file(filename):
    """
    Load a PyTree of parameters from a .safetensors file.

    This expects that the file was written with `safetensors.flax.save_file`
    on a parameter pytree that matches the skill Q-network params used in training
    (typically something like `train_states.params` from the trainer).
    """
    resolved = resolve_config_path(filename)
    print(f"Loading weights from: {resolved}")
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"Checkpoint not found: {resolved}\n"
            f"  (config path was: {filename!r})\n"
            "  Train skills first (e.g. TRAIN_MODE=skills_only) so SAVE_PATH writes the "
            ".safetensors file, or set INIT_MODEL_PATH to an existing file."
        )
    flattened_dict = load_file(resolved)
    return unflatten_dict(flattened_dict, sep=",")


def _checkpoint_base_stem(stem: str) -> str:
    """Strip a skills suffix so sibling meta/config files share the same base name."""
    if stem.endswith("_skills"):
        return stem[: -len("_skills")]
    return stem


def find_checkpoint_config_path(model_path: str) -> str | None:
    """
    Locate the training config saved next to a checkpoint by main.py
    (``{base}_config.yaml``).
    """
    resolved = resolve_config_path(model_path)
    if resolved is None:
        return None
    candidate = Path(resolved).parent / f"{_checkpoint_base_stem(Path(resolved).stem)}_config.yaml"
    return str(candidate) if candidate.is_file() else None


def _infer_env_name_from_checkpoint_path(model_path: str) -> str | None:
    """Guess ENV_NAME from ``.../outputs/models/<env>/...`` layout used by main.py."""
    resolved = resolve_config_path(model_path)
    if resolved is None:
        return None
    parent = Path(resolved).parent.name
    known = {"seaquest", "phoenix", "choppercommand", "spaceinvaders"}
    return parent if parent in known else None


def load_config_for_checkpoint(model_path: str, base_config: dict) -> tuple[dict, bool]:
    """Merge a saved training config over the visualizer defaults when available."""
    cfg_path = find_checkpoint_config_path(model_path)
    if cfg_path is not None:
        exp_cfg = OmegaConf.load(cfg_path)
        exp_dict = OmegaConf.to_container(exp_cfg, resolve=True)
        merged = {**base_config, **exp_dict, **exp_dict.get("alg", {})}
        print(f"Loaded training config from: {cfg_path}")
        return merged, True

    env_guess = _infer_env_name_from_checkpoint_path(model_path)
    if env_guess is not None and base_config.get("ENV_NAME") != env_guess:
        print(
            f"No *_config.yaml next to checkpoint; using ENV_NAME={env_guess!r} "
            f"from parent directory (was {base_config.get('ENV_NAME')!r})."
        )
        return {**base_config, "ENV_NAME": env_guess}, False
    return base_config, False


def resolve_tandem_checkpoint_paths(
    skill_path: str | None,
    meta_path: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Resolve skill and meta checkpoint paths for tandem runs.

    Tandem training writes ``*_skills.safetensors`` and ``*_meta.safetensors``.
    When only the skills file is passed, auto-load the sibling meta file if present.
    """
    if skill_path is None:
        return None, resolve_config_path(meta_path) if meta_path else None
    skill_resolved = resolve_config_path(skill_path)
    if meta_path is not None:
        return skill_resolved, resolve_config_path(meta_path)
    meta_candidate = (
        Path(skill_resolved).parent
        / f"{_checkpoint_base_stem(Path(skill_resolved).stem)}_meta.safetensors"
    )
    if meta_candidate.is_file():
        print(f"Auto-detected meta checkpoint: {meta_candidate}")
        return skill_resolved, str(meta_candidate)
    return skill_resolved, None


def _unwrap_stacked_skill_params(params_tree):
    """Return a single-skill param subtree and the number of stacked skills."""
    kernel = params_tree["Dense_0"]["kernel"]
    arr = np.asarray(kernel)
    if arr.ndim == 3:
        num_skills = int(arr.shape[0])
        single = jax.tree_util.tree_map(lambda x: x[0], params_tree)
        return single, num_skills
    return params_tree, 1


def infer_skill_hparams_from_params(params_tree) -> dict:
    """
    Infer skill Q-network hyperparameters from a checkpoint parameter tree.

    Expects Flax ``QNetwork`` param keys ``Dense_0`` … ``Dense_{NUM_LAYERS}``.
    """
    sample, num_skills = _unwrap_stacked_skill_params(params_tree)
    dense_indices = sorted(
        int(key.split("_", 1)[1])
        for key in sample
        if key.startswith("Dense_") and key.split("_", 1)[1].isdigit()
    )
    if not dense_indices:
        raise ValueError("Skill checkpoint has no Dense_* layers; cannot infer architecture.")

    first = sample[f"Dense_{dense_indices[0]}"]["kernel"]
    last = sample[f"Dense_{dense_indices[-1]}"]["kernel"]
    ego_in = int(first.shape[0])
    hidden_size = int(first.shape[1])
    ego_action_dim = int(last.shape[1])
    num_layers = len(dense_indices) - 1

    channels_per_frame = ego_obs_channels_per_frame()
    frame_stack_size = (
        ego_in // channels_per_frame if ego_in % channels_per_frame == 0 else None
    )

    return {
        "HIDDEN_SIZE": hidden_size,
        "NUM_LAYERS": num_layers,
        "EGO_ACTION_DIM": ego_action_dim,
        "NUM_SKILLS": num_skills,
        "FRAME_STACK_SIZE": frame_stack_size,
        "_ckpt_ego_in": ego_in,
    }


def ensure_skill_params_stack(params_tree, num_skills: int):
    """Ensure skill params are stacked along axis 0 for ``jax.vmap`` over skills."""
    kernel = np.asarray(params_tree["Dense_0"]["kernel"])
    if kernel.ndim == 2:
        return jax.tree_util.tree_map(
            lambda p: jnp.stack([p] * num_skills, axis=0),
            params_tree,
        )
    return params_tree


def align_config_with_skill_checkpoint(config: dict, params_tree) -> dict:
    """Override skill architecture fields so the network matches the checkpoint."""
    inferred = infer_skill_hparams_from_params(params_tree)
    ckpt_in = inferred.pop("_ckpt_ego_in")
    frame_stack_size = inferred.get("FRAME_STACK_SIZE")
    updated = {**config, **{k: v for k, v in inferred.items() if v is not None}}
    if int(config.get("HIDDEN_SIZE", -1)) != int(updated["HIDDEN_SIZE"]):
        print(
            f"Using checkpoint HIDDEN_SIZE={updated['HIDDEN_SIZE']} "
            f"(default config had {config.get('HIDDEN_SIZE')})."
        )
    if frame_stack_size is not None and int(
        config.get("FRAME_STACK_SIZE", frame_stack_size)
    ) != int(frame_stack_size):
        print(
            f"Using checkpoint FRAME_STACK_SIZE={frame_stack_size} "
            f"(default config had {config.get('FRAME_STACK_SIZE')})."
        )
    updated["_ckpt_ego_in"] = ckpt_in
    return updated


def validate_skill_checkpoint_ego_dim(config: dict, ego_feat_dim: int) -> None:
    """Fail fast when the eval env's ego observation length does not match the checkpoint."""
    ckpt_in = config.pop("_ckpt_ego_in", None)
    if ckpt_in is None:
        return
    if int(ego_feat_dim) == int(ckpt_in):
        return
    channels_per_frame = ego_obs_channels_per_frame()
    need_stack = (
        ckpt_in // channels_per_frame
        if ckpt_in % channels_per_frame == 0
        else "?"
    )
    raise ValueError(
        f"Skill checkpoint expects MLP input size {ckpt_in}, but this environment's "
        f"egocentric observation length is {ego_feat_dim}. "
        f"Ego length is {channels_per_frame} × frame stack. Set FRAME_STACK_SIZE to "
        f"{need_stack} (and ENV_NAME to the game the checkpoint was trained on), or pass "
        f"the matching *_config.yaml via a checkpoint saved by main.py. Also ensure "
        "USE_LEGACY_EGO_OBS / USE_EGO_SIZE_FEATURES / USE_EGO_VELOCITY_FEATURES "
        "match the checkpoint's training configuration."
    )


def create_eval_env(config):
    """
    Create the evaluation environment + renderer using the same semantic
    wrappers as the training pipeline, selected via the EnvManager.

    The choice of environment (Seaquest vs others) is driven by
    `config["ENV_NAME"]`, which can be overridden on the command line.
    """
    env_name = config.get("ENV_NAME", "Seaquest")
    manager = EnvManager()
    env, renderer = manager.make_env(env_name, config, train=False)
    return env, renderer


def label_to_skill_idx(label, num_skills: int):
    """
    Map semantic labels to skill indices, matching the convention in train_agent.make_train:
    0 = NOOP, 1 = DESTROY, 2 = COLLECT, 3 = AVOID  → skills 0..num_skills-1 (no NOOP skill).
    """
    return jnp.clip(label - 1, 0, num_skills - 1)


def _select_target_with_meta(
    obs,
    config,
    meta_network,
    meta_params,
    meta_mask_hook,
    forced_target_hook,
    no_target_idx,
):
    """Select a target index using a learned meta-policy checkpoint."""
    meta_q_vals = meta_network.apply(
        {"params": meta_params},
        obs.raw_observation,
        obs.objects,
        train=False,
    )
    active_mask = build_meta_target_mask(
        obs.objects,
        jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), obs.raw_observation),
        config,
        hook=meta_mask_hook,
    )
    if active_mask.ndim > 1:
        active_mask = active_mask[0]

    any_active = jnp.any(active_mask)
    valid_qs = jnp.where(active_mask, meta_q_vals, -jnp.inf)
    target_idx = jnp.argmax(valid_qs)
    target_idx = forced_target_hook(
        jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), obs.raw_observation),
        jnp.expand_dims(target_idx, 0),
        config,
    )[0]
    target_idx = jnp.where(any_active, target_idx, no_target_idx)
    return int(target_idx)


def run_greedy_eval_episodes(
    config,
    params_tree=None,
    meta_params=None,
    *,
    num_episodes: int = 5,
    num_seeds: int = 3,
    base_seed: int | None = None,
    max_frames: int | None = None,
) -> dict:
    """
    Greedy hierarchical eval on the fair eval wrapper (no sticky actions, no episodic life).

    Runs ``num_seeds * num_episodes`` full episodes and returns aggregate score stats.
    Meta is greedy argmax when ``meta_params`` is set; otherwise the env heuristic is used.
    """
    set_ego_obs_format(bool(config.get("USE_LEGACY_EGO_OBS", False)))
    set_ego_size_features(bool(config.get("USE_EGO_SIZE_FEATURES", True)))
    set_ego_velocity_features(bool(config.get("USE_EGO_VELOCITY_FEATURES", True)))
    clip_px = config.get("EGO_DISTANCE_CLIP_MAX_PX", None)
    set_ego_distance_clip_max_pixels(
        None if clip_px is None else float(clip_px)
    )

    num_skills = int(config["NUM_SKILLS"])
    ego_action_dim = int(config["EGO_ACTION_DIM"])
    hidden_size = int(config["HIDDEN_SIZE"])
    num_layers = int(config["NUM_LAYERS"])
    base_seed = int(config["SEED"] if base_seed is None else base_seed)
    restrict_fire_for_non_destroy = bool(
        config.get("RESTRICT_FIRE_FOR_NON_DESTROY", False)
    )
    meta_use_noop_target = bool(config.get("META_USE_NOOP_TARGET", False))
    meta_noop_target_idx = int(config.get("META_NOOP_TARGET_IDX", 0))
    meta_use_fire_target = bool(config.get("META_USE_FIRE_TARGET", False))
    meta_fire_target_idx = int(config.get("META_FIRE_TARGET_IDX", 1))
    meta_fire_game_action = int(config.get("META_FIRE_GAME_ACTION", 1))
    max_frames = int(max_frames or config.get("EVAL_MAX_FRAMES", 27000))

    env, _renderer = create_eval_env(config)
    env_name = config.get("ENV_NAME", "Seaquest")
    manager = EnvManager()
    components = manager.get_components(env_name)
    heuristic_target = components.get_heuristic_target
    no_target_idx = components.no_target_idx
    action_mapper = components.action_mapper
    forced_target_hook = components.forced_target_hook
    meta_mask_hook = components.meta_mask_hook

    skill_network = QNetwork(
        action_dim=ego_action_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    meta_network = create_meta_network(config) if meta_params is not None else None

    rng = jax.random.PRNGKey(base_seed)
    obs, _state = env.reset(rng)
    dummy_ego = get_ego_obs(
        obs.raw_observation,
        obs.objects,
        jnp.array(0, dtype=jnp.int32),
    )
    validate_skill_checkpoint_ego_dim(config, int(dummy_ego.shape[-1]))
    if params_tree is not None:
        params_tree = ensure_skill_params_stack(params_tree, num_skills)

    episode_returns: list[float] = []
    for seed_idx in range(num_seeds):
        for ep_idx in range(num_episodes):
            ep_seed = base_seed + seed_idx * 10_000 + ep_idx * 997
            rng = jax.random.PRNGKey(ep_seed)
            rng, rng_reset = jax.random.split(rng)
            obs, state = env.reset(rng_reset)
            prev_target_idx = no_target_idx
            episode_return = 0.0

            for _ in range(max_frames):
                rng, step_key = jax.random.split(rng)

                if meta_network is None:
                    target_obj_idx = int(
                        jnp.ravel(
                            heuristic_target(
                                obs.raw_observation,
                                obs.objects,
                                prev_target_idx,
                                step_key,
                            )
                        )[0]
                    )
                    target_obj_idx = int(
                        forced_target_hook(
                            jax.tree_util.tree_map(
                                lambda x: jnp.expand_dims(x, 0), obs.raw_observation
                            ),
                            jnp.asarray([target_obj_idx], dtype=jnp.int32),
                            config,
                        )[0]
                    )
                else:
                    target_obj_idx = _select_target_with_meta(
                        obs,
                        config,
                        meta_network,
                        meta_params,
                        meta_mask_hook,
                        forced_target_hook,
                        no_target_idx,
                    )
                prev_target_idx = target_obj_idx

                is_meta_noop = (
                    meta_use_noop_target and target_obj_idx == meta_noop_target_idx
                )
                is_meta_fire = (
                    meta_use_fire_target and target_obj_idx == meta_fire_target_idx
                )
                if target_obj_idx == no_target_idx or is_meta_noop:
                    global_action = 0
                elif is_meta_fire:
                    global_action = meta_fire_game_action
                else:
                    raw_label = obs.objects.semantic_label[-1, target_obj_idx]
                    active_skill_idx = int(label_to_skill_idx(raw_label, num_skills))
                    ego_obs = get_ego_obs(
                        obs.raw_observation,
                        obs.objects,
                        jnp.array(target_obj_idx, dtype=jnp.int32),
                    )
                    ego_obs_batch = ego_obs[jnp.newaxis, ...]

                    def apply_one(params_for_skill):
                        return skill_network.apply(
                            {"params": params_for_skill},
                            ego_obs_batch,
                            train=False,
                        )

                    q_vals = jax.vmap(apply_one)(params_tree)
                    q_vals = jnp.squeeze(q_vals, axis=1)
                    q_active = q_vals[active_skill_idx]
                    if restrict_fire_for_non_destroy and active_skill_idx != 0:
                        fire_indices = fire_ego_action_indices(ego_action_dim)
                        q_active = q_active.at[fire_indices].set(-jnp.inf)
                    chosen_ego = int(jnp.argmax(q_active))
                    global_action = int(
                        action_mapper(
                            chosen_ego,
                            obs.raw_observation,
                            jnp.array(target_obj_idx, dtype=jnp.int32),
                            obs.objects,
                        )
                    )

                obs, state, reward, done, _info = env.step(state, global_action)
                episode_return += float(reward)
                if done:
                    break

            episode_returns.append(episode_return)

    arr = np.asarray(episode_returns, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std()) if arr.size > 1 else 0.0,
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "returns": episode_returns,
        "num_episodes": int(arr.size),
        "num_seeds": int(num_seeds),
        "episodes_per_seed": int(num_episodes),
    }


def run_eval_rollout(config, params_tree=None, meta_params=None, max_frames=None):
    """
    Run a single evaluation rollout with the hierarchical PQN-style agent:
    - Fixed heuristic meta-policy to choose a target object.
    - Shared skill Q-network (one set of weights per skill).
    - Ego-centric action space mapped back to Atari actions.

    Returns the same tensors that `video_recorder.video_callback` expects.
    """
    set_ego_obs_format(bool(config.get("USE_LEGACY_EGO_OBS", False)))
    set_ego_size_features(bool(config.get("USE_EGO_SIZE_FEATURES", True)))
    set_ego_velocity_features(bool(config.get("USE_EGO_VELOCITY_FEATURES", True)))
    clip_px = config.get("EGO_DISTANCE_CLIP_MAX_PX", None)
    set_ego_distance_clip_max_pixels(
        None if clip_px is None else float(clip_px)
    )
    num_skills = config["NUM_SKILLS"]
    ego_action_dim = config["EGO_ACTION_DIM"]
    hidden_size = config["HIDDEN_SIZE"]
    num_layers = config["NUM_LAYERS"]
    seed = config["SEED"]
    restrict_fire_for_non_destroy = bool(
        config.get("RESTRICT_FIRE_FOR_NON_DESTROY", False)
    )
    meta_use_noop_target = bool(config.get("META_USE_NOOP_TARGET", False))
    meta_noop_target_idx = int(config.get("META_NOOP_TARGET_IDX", 0))
    meta_use_fire_target = bool(config.get("META_USE_FIRE_TARGET", False))
    meta_fire_target_idx = int(config.get("META_FIRE_TARGET_IDX", 1))
    meta_fire_game_action = int(config.get("META_FIRE_GAME_ACTION", 1))
    restrict_fire_for_non_destroy = bool(
        config.get("RESTRICT_FIRE_FOR_NON_DESTROY", False)
    )
    max_frames = max_frames or config.get("VIDEO_MAX_FRAMES", 2000)

    env, renderer = create_eval_env(config)

    # Environment-specific hooks (meta-policy, action mapping, NO_TARGET sentinel)
    env_name = config.get("ENV_NAME", "Seaquest")
    manager = EnvManager()
    components = manager.get_components(env_name)
    heuristic_target = components.get_heuristic_target
    no_target_idx = components.no_target_idx
    action_mapper = components.action_mapper
    forced_target_hook = components.forced_target_hook
    meta_mask_hook = components.meta_mask_hook

    meta_network = None
    if meta_params is not None:
        meta_network = create_meta_network(config)

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
    validate_skill_checkpoint_ego_dim(config, ego_feat_dim)
    if params_tree is not None:
        params_tree = ensure_skill_params_stack(params_tree, int(config["NUM_SKILLS"]))

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
    prev_target_idx = no_target_idx
    for _ in range(max_frames):
        rng, step_key = jax.random.split(rng)

        if meta_network is None:
            # Heuristic meta-policy selects a target object
            target_obj_idx = heuristic_target(
                obs.raw_observation,
                obs.objects,
                prev_target_idx,
                step_key,
            )
            target_obj_idx = int(jnp.ravel(target_obj_idx)[0])
            target_obj_idx = int(
                forced_target_hook(
                    jax.tree_util.tree_map(
                        lambda x: jnp.expand_dims(x, 0), obs.raw_observation
                    ),
                    jnp.asarray([target_obj_idx], dtype=jnp.int32),
                    config,
                )[0]
            )
        else:
            target_obj_idx = _select_target_with_meta(
                obs,
                config,
                meta_network,
                meta_params,
                meta_mask_hook,
                forced_target_hook,
                no_target_idx,
            )
        prev_target_idx = target_obj_idx

        is_meta_noop = meta_use_noop_target and target_obj_idx == meta_noop_target_idx
        is_meta_fire = meta_use_fire_target and target_obj_idx == meta_fire_target_idx
        if target_obj_idx == no_target_idx or is_meta_noop:
            global_action = 0
            active_skill_idx = 0
            combined_q = np.zeros(num_skills, dtype=np.float32)
            distance = np.nan
            angle = np.nan
        elif is_meta_fire:
            global_action = meta_fire_game_action
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
            # See `get_ego_obs` / `EGO_OBS_CHANNELS_PER_FRAME` for the per-frame layout.
            channels_per_frame = ego_obs_channels_per_frame()
            last_frame_ego = ego_obs[-channels_per_frame:]
            if using_legacy_ego_obs():
                dx_pix = float(last_frame_ego[0] * SCREEN_WIDTH)
                dy_pix = float(last_frame_ego[1] * SCREEN_HEIGHT)
                distance = float(jnp.sqrt(dx_pix ** 2 + dy_pix ** 2))
                angle = float(jnp.arctan2(dy_pix, dx_pix))
            else:
                distance = distance_pixels_from_feature(float(last_frame_ego[0]))
                player = _get_player(obs.raw_observation)
                y_is_bottom_edge = _uses_bottom_edge_y(obs.raw_observation)
                p_cx = float(
                    player.x[-1]
                    + player.width[-1] / 2.0
                )
                p_cy = float(
                    _center_y(
                        player.y[-1],
                        player.height[-1],
                        y_is_bottom_edge=y_is_bottom_edge,
                    )
                )
                t_cx = float(
                    obs.objects.x[-1, target_obj_idx]
                    + obs.objects.width[-1, target_obj_idx] / 2.0
                )
                t_cy = float(
                    _center_y(
                        obs.objects.y[-1, target_obj_idx],
                        obs.objects.height[-1, target_obj_idx],
                        y_is_bottom_edge=y_is_bottom_edge,
                    )
                )
                angle = float(jnp.arctan2(t_cy - p_cy, t_cx - p_cx))

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

            fire_indices = fire_ego_action_indices(ego_action_dim)
            q_active = q_vals[active_skill_idx]
            if restrict_fire_for_non_destroy and active_skill_idx != 0:
                q_active = q_active.at[fire_indices].set(-jnp.inf)

            chosen_ego = int(jnp.argmax(q_active))
            global_action = int(
                action_mapper(
                    chosen_ego,
                    obs.raw_observation,
                    jnp.array(target_obj_idx, dtype=jnp.int32),
                    obs.objects,
                )
            )
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


def run_live_visualization(
    config,
    params_tree=None,
    meta_params=None,
    max_frames=None,
    fps: int | None = None,
):
    """
    Run a live, interactive visualization loop using pygame.
    The agent acts in the environment step-by-step and the current frame is
    rendered to a window, with the skill input (distance, angle) shown at the bottom.
    """
    set_ego_obs_format(bool(config.get("USE_LEGACY_EGO_OBS", False)))
    set_ego_size_features(bool(config.get("USE_EGO_SIZE_FEATURES", True)))
    set_ego_velocity_features(bool(config.get("USE_EGO_VELOCITY_FEATURES", True)))
    clip_px = config.get("EGO_DISTANCE_CLIP_MAX_PX", None)
    set_ego_distance_clip_max_pixels(
        None if clip_px is None else float(clip_px)
    )
    num_skills = config["NUM_SKILLS"]
    ego_action_dim = config["EGO_ACTION_DIM"]
    hidden_size = config["HIDDEN_SIZE"]
    num_layers = config["NUM_LAYERS"]
    seed = config["SEED"]
    restrict_fire_for_non_destroy = bool(
        config.get("RESTRICT_FIRE_FOR_NON_DESTROY", False)
    )
    meta_use_noop_target = bool(config.get("META_USE_NOOP_TARGET", False))
    meta_noop_target_idx = int(config.get("META_NOOP_TARGET_IDX", 0))
    meta_use_fire_target = bool(config.get("META_USE_FIRE_TARGET", False))
    meta_fire_target_idx = int(config.get("META_FIRE_TARGET_IDX", 1))
    meta_fire_game_action = int(config.get("META_FIRE_GAME_ACTION", 1))
    # VIDEO_MAX_FRAMES is for recorded rollouts; live pygame runs until quit unless capped.
    frame_limit = max_frames
    display_fps = int(fps if fps is not None else config.get("VISUALIZER_FPS", 15))

    env, renderer = create_eval_env(config)

    # Environment-specific hooks (meta-policy, action mapping, NO_TARGET sentinel)
    env_name = config.get("ENV_NAME", "Seaquest")
    manager = EnvManager()
    components = manager.get_components(env_name)
    heuristic_target = components.get_heuristic_target
    no_target_idx = components.no_target_idx
    action_mapper = components.action_mapper
    forced_target_hook = components.forced_target_hook
    meta_mask_hook = components.meta_mask_hook

    meta_network = None
    if meta_params is not None:
        meta_network = create_meta_network(config)

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
    validate_skill_checkpoint_ego_dim(config, ego_feat_dim)
    if params_tree is not None:
        params_tree = ensure_skill_params_stack(params_tree, int(config["NUM_SKILLS"]))

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
    pygame.display.set_caption("Hierarchical PQN Live Visualization")
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
    prev_target_idx = no_target_idx
    frame_count = 0

    def reset_env_episode():
        nonlocal obs, state, prev_target_idx, rng
        rng_local, rng_reset_local = jax.random.split(rng)
        obs_local, state_local = env.reset(rng_reset_local)
        obs = obs_local
        state = state_local
        rng = rng_local
        prev_target_idx = no_target_idx

    while running and (frame_limit is None or frame_count < frame_limit):
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                reset_env_episode()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                # Snapshot the current heuristic view of the scene.
                # This is especially useful for ChopperCommand debugging, but will run
                # for any env that provides compatible observations.
                snap = debug_heuristic_snapshot(
                    obs.raw_observation,
                    obs.objects,
                    jnp.array(prev_target_idx, dtype=jnp.int32),
                )

        rng, step_key = jax.random.split(rng)

        if meta_network is None:
            # Heuristic meta-policy selects a target object
            target_obj_idx = heuristic_target(
                obs.raw_observation,
                obs.objects,
                prev_target_idx,
                step_key,
            )
            target_obj_idx = int(jnp.ravel(target_obj_idx)[0])
            target_obj_idx = int(
                forced_target_hook(
                    jax.tree_util.tree_map(
                        lambda x: jnp.expand_dims(x, 0), obs.raw_observation
                    ),
                    jnp.asarray([target_obj_idx], dtype=jnp.int32),
                    config,
                )[0]
            )
        else:
            target_obj_idx = _select_target_with_meta(
                obs,
                config,
                meta_network,
                meta_params,
                meta_mask_hook,
                forced_target_hook,
                no_target_idx,
            )
        prev_target_idx = target_obj_idx

        is_meta_noop = meta_use_noop_target and target_obj_idx == meta_noop_target_idx
        is_meta_fire = meta_use_fire_target and target_obj_idx == meta_fire_target_idx
        if target_obj_idx == no_target_idx or is_meta_noop:
            global_action = 0
            active_skill_idx = 0
            distance = np.nan
            angle = np.nan
        elif is_meta_fire:
            global_action = meta_fire_game_action
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

            channels_per_frame = ego_obs_channels_per_frame()
            last_frame_ego = ego_obs[-channels_per_frame:]
            if using_legacy_ego_obs():
                dx_pix = float(last_frame_ego[0] * SCREEN_WIDTH)
                dy_pix = float(last_frame_ego[1] * SCREEN_HEIGHT)
                distance = float(jnp.sqrt(dx_pix ** 2 + dy_pix ** 2))
                angle = float(jnp.arctan2(dy_pix, dx_pix))
            else:
                distance = distance_pixels_from_feature(float(last_frame_ego[0]))
                player = _get_player(obs.raw_observation)
                y_is_bottom_edge = _uses_bottom_edge_y(obs.raw_observation)
                p_cx = float(
                    player.x[-1]
                    + player.width[-1] / 2.0
                )
                p_cy = float(
                    _center_y(
                        player.y[-1],
                        player.height[-1],
                        y_is_bottom_edge=y_is_bottom_edge,
                    )
                )
                t_cx = float(
                    obs.objects.x[-1, target_obj_idx]
                    + obs.objects.width[-1, target_obj_idx] / 2.0
                )
                t_cy = float(
                    _center_y(
                        obs.objects.y[-1, target_obj_idx],
                        obs.objects.height[-1, target_obj_idx],
                        y_is_bottom_edge=y_is_bottom_edge,
                    )
                )
                angle = float(jnp.arctan2(t_cy - p_cy, t_cx - p_cx))

            def apply_one(params_for_skill):
                return skill_network.apply(
                    {"params": params_for_skill},
                    ego_obs_batch,
                    train=False,
                )

            q_vals = jax.vmap(apply_one)(params_tree)
            q_vals = jnp.squeeze(q_vals, axis=1)

            fire_indices = fire_ego_action_indices(ego_action_dim)
            q_active = q_vals[active_skill_idx]
            if restrict_fire_for_non_destroy and active_skill_idx != 0:
                q_active = q_active.at[fire_indices].set(-jnp.inf)

            chosen_ego = int(jnp.argmax(q_active))
            global_action = int(
                action_mapper(
                    chosen_ego,
                    obs.raw_observation,
                    jnp.array(target_obj_idx, dtype=jnp.int32),
                    obs.objects,
                )
            )
        # Step env
        obs, state, _reward, done, _info = env.step(state, global_action)

        # Render current frame
        base_state = state.atari_state.env_state
        frame = renderer.render(base_state)
        frame = np.array(jax.device_get(frame), dtype=np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        # Add a red border around the selected target object (same style as videos).
        if target_obj_idx != no_target_idx:
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
            text = f"skill: d={distance:.1f}px, a={angle:.2f} rad"
            text_surface = font.render(text, True, (255, 255, 255))
            tx = (window_size[0] - text_surface.get_width()) // 2
            ty = window_size[1] - text_surface.get_height() - 10
            screen.blit(text_surface, (tx, ty))

        pygame.display.flip()
        clock.tick(display_fps)
        frame_count += 1

        if done:
            rng, rng_reset = jax.random.split(rng)
            obs, state = env.reset(rng_reset)
            prev_target_idx = no_target_idx

    if frame_limit is not None and frame_count >= frame_limit:
        print(f"Live visualization stopped after {frame_count} frames (--max-frames={frame_limit}).")
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
        help=(
            "Cap total frames. Record mode defaults to VIDEO_MAX_FRAMES from config (600). "
            "Live pygame mode runs until you close the window unless this is set."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Live window refresh rate in frames per second (default: 15, or VISUALIZER_FPS in config).",
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
        default="base",
        help="Which Seaquest modpack to use: one of 'base', 'collect', 'destroy', or 'avoid'.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Record a video of the agent playing the environment.",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=None,
        help=(
            "Override ENV_NAME from the loaded config. "
            "Determines which environment wrapper, meta-policy, and action "
            "mapping are used by the visualizer."
        ),
    )
    parser.add_argument(
        "--meta-model",
        type=str,
        default=None,
        help=(
            "Optional path to a learned meta-policy checkpoint "
            "(*_meta.safetensors). If set, target selection uses MetaQNetwork "
            "instead of the heuristic meta-policy. When loading a tandem "
            "*_skills.safetensors file, the sibling *_meta.safetensors is "
            "auto-detected if present."
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=(
            "Override the experiment yaml from configs/config.yaml defaults "
            "(e.g. phoenix_tandem, phoenix_skills). Ignored when a *_config.yaml "
            "is found next to the checkpoint."
        ),
    )
    args = parser.parse_args()

    base_config = load_base_config(experiment=args.experiment)

    # Optionally override the environment used for visualization.
    if args.env_name is not None:
        base_config["ENV_NAME"] = args.env_name
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

    # Optional parameter loading (skills + optional tandem meta head)
    params_tree = None
    meta_params = None
    skill_path, meta_path = resolve_tandem_checkpoint_paths(args.model, args.meta_model)
    if skill_path is not None:
        base_config, _ = load_config_for_checkpoint(skill_path, base_config)
        params_tree = load_params_from_file(skill_path)
        base_config = align_config_with_skill_checkpoint(base_config, params_tree)
    if meta_path is not None:
        if meta_params is None:
            base_config, _ = load_config_for_checkpoint(meta_path, base_config)
        meta_params = load_params_from_file(meta_path)

    # Keep explicit CLI env override authoritative for cross-env transfer runs,
    # even when checkpoint-side *_config.yaml is present.
    if args.env_name is not None:
        base_config["ENV_NAME"] = args.env_name

    # If requested, run live interactive visualization and exit.
    if not args.record:
        run_live_visualization(
            base_config,
            params_tree=params_tree,
            meta_params=meta_params,
            max_frames=args.max_frames,
            fps=args.fps,
        )
        return

    # Initialize wandb so that `video_recorder` can log the video (offline by default).
    if base_config.get("WANDB_MODE", "disabled") != "disabled":
        wandb.init(
            entity=base_config.get("ENTITY", "paul-seitz-tu-darmstadt"),
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
    ) = run_eval_rollout(
        base_config,
        params_tree=params_tree,
        meta_params=meta_params,
        max_frames=args.max_frames,
    )

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