import os
import time
import copy
import functools
import warnings
import jax
import jax.numpy as jnp
import wandb
from flax.traverse_util import flatten_dict
from safetensors.flax import save_file
import hydra
from omegaconf import OmegaConf

from concept_based_learning.environments.env_manager import EnvManager
from concept_based_learning.environments.heuristics import get_nearest_object_target
from concept_based_learning.curriculum import build_skill_phases
from concept_based_learning.training.train_skills import make_train
from concept_based_learning._legacy.train_meta_agent import make_meta_train
from concept_based_learning.training.train_tandem import make_tandem_train
from concept_based_learning.training.rtpt_progress import start_rtpt
from concept_based_learning.observations.egocentric import (
    set_ego_obs_format,
    set_ego_size_features,
    set_ego_velocity_features,
    set_ego_distance_clip_max_pixels,
    ego_obs_channels_per_frame,
)
from concept_based_learning.video_recorder import video_callback, wait_for_video_finish
from concept_based_learning.pqn_visualizer import (
    load_params_from_file,
    resolve_config_path,
    run_eval_rollout,
    infer_skill_hparams_from_params,
)


def _safe_checkpoint_path(base_path, init_model_path=None):
    """
    Returns a safe output path that never overwrites `init_model_path`.
    If collision is detected, append `_finetuned` before extension.
    """
    if init_model_path is None:
        return base_path

    try:
        base_abs = os.path.abspath(base_path)
        init_abs = os.path.abspath(init_model_path)
    except Exception:
        return base_path

    if base_abs != init_abs:
        return base_path

    root, ext = os.path.splitext(base_path)
    return f"{root}_finetuned{ext}"


def _configure_ego_obs_format(config: dict) -> None:
    """Select ego observation format and feature toggles for this run."""
    set_ego_obs_format(bool(config.get("USE_LEGACY_EGO_OBS", False)))
    set_ego_size_features(bool(config.get("USE_EGO_SIZE_FEATURES", True)))
    set_ego_velocity_features(bool(config.get("USE_EGO_VELOCITY_FEATURES", True)))
    clip_px = config.get("EGO_DISTANCE_CLIP_MAX_PX", None)
    set_ego_distance_clip_max_pixels(
        None if clip_px is None else float(clip_px)
    )
    stack = int(config.get("FRAME_STACK_SIZE", 4))
    ch = ego_obs_channels_per_frame()
    clip_note = (
        f"distance_clip={clip_px}px"
        if clip_px is not None
        else "distance_clip=off"
    )
    print(
        f"Ego observations: {ch} channels/frame × {stack} stack = {ch * stack} "
        f"(legacy={config.get('USE_LEGACY_EGO_OBS', False)}, "
        f"size={config.get('USE_EGO_SIZE_FEATURES', True)}, "
        f"velocity={config.get('USE_EGO_VELOCITY_FEATURES', True)}, "
        f"{clip_note})"
    )


def outer_make_train(config, frozen_meta_params=None):
    """
    Constructs the environments with the custom semantic wrappers defined
    in `EnvManager` and passes them to the training factory.

    Eval / test env uses the same `GAME_MODS` as training by default (when
    `EVAL_GAME_MODS` is null or omitted) so end-of-phase videos and eval
    rollouts match the environment actually trained on (e.g. curriculum mods).

    Skillbootcamp: set ``EGO_ACTION_DIM`` to **18** (full egocentric table) or **6**
    (primitives only) so skill Q-heads match ``bootcamp_action_mapper``; the mapper
    supports both (see ``bootcamp_action_mapper`` module docstring).
    """
    env_name = str(config.get("ENV_NAME", "Seaquest"))
    _configure_ego_obs_format(config)

    manager = EnvManager()
    components = manager.get_components(env_name)
    # Default: eval inherits training `GAME_MODS` (null / absent). Set
    # `EVAL_GAME_MODS` to a non-empty list to override eval only; use `[]`
    # if you need an unmodified env for eval while training with mods.
    train_game_mods = config.get("GAME_MODS", None)
    raw_eval_game_mods = config.get("EVAL_GAME_MODS", None)
    eval_game_mods = train_game_mods if raw_eval_game_mods is None else raw_eval_game_mods

    def create_env_with_mods(game_mods, train: bool):
        local_cfg = dict(config)
        if game_mods is not None:
            local_cfg["GAME_MODS"] = game_mods
        else:
            local_cfg.pop("GAME_MODS", None)
        env, renderer = manager.make_env(env_name, local_cfg, train=train)
        return env, renderer

    env, _ = create_env_with_mods(train_game_mods, train=True)
    test_env, renderer = create_env_with_mods(eval_game_mods, train=False)

    # Match tandem: optional env-agnostic nearest-object heuristic vs env-specific policy.
    heuristic_mode = config.get("HEURISTIC_MODE", "env_specific")
    if heuristic_mode == "nearest":
        heuristic_fn = functools.partial(
            get_nearest_object_target, no_target_idx=components.no_target_idx
        )
    else:
        heuristic_fn = components.get_heuristic_target

    # Call the hierarchical training loop; returns (train_fn, eval_rollout_fn)
    train_fn, eval_rollout_fn = make_train(
        config,
        env,
        test_env,
        renderer,
        frozen_meta_params=frozen_meta_params,
        get_heuristic_target=heuristic_fn,
        no_target_idx=components.no_target_idx,
        action_mapper=components.action_mapper,
        inverse_action_mapper=components.inverse_action_mapper,
        extract_all_skills_rewards=components.reward_extractor,
        extract_life_lost=components.extract_life_lost,
        fixed_target_idx_from_config=components.fixed_target_idx_from_config,
        meta_mask_hook=components.meta_mask_hook,
        forced_target_hook=components.forced_target_hook,
    )
    return train_fn, eval_rollout_fn, renderer


def outer_make_meta_train(config, frozen_skill_params):
    """
    Constructs base environments and returns the learned meta-policy trainer.
    Meta training is intentionally forced onto the unmodified base environment.
    """
    env_name = str(config.get("ENV_NAME", "Seaquest"))
    _configure_ego_obs_format(config)

    manager = EnvManager()
    components = manager.get_components(env_name)

    local_cfg = dict(config)
    # Strict guarantee: never let curriculum/environment mods leak into meta training.
    local_cfg.pop("GAME_MODS", None)
    local_cfg.pop("EVAL_GAME_MODS", None)

    env, _ = manager.make_env(env_name, local_cfg, train=True)
    test_env, renderer = manager.make_env(env_name, local_cfg, train=False)

    train_fn, eval_rollout_fn = make_meta_train(
        local_cfg,
        env,
        test_env,
        renderer,
        frozen_skill_params=frozen_skill_params,
        no_target_idx=components.no_target_idx,
        action_mapper=components.action_mapper,
        extract_life_lost=components.extract_life_lost,
        meta_mask_hook=components.meta_mask_hook,
        forced_target_hook=components.forced_target_hook,
    )
    return train_fn, eval_rollout_fn, renderer


def outer_make_tandem_train(config):
    """
    Constructs base environments and returns the tandem trainer.
    Tandem training runs on the configured base environment without curriculum phases.
    """
    env_name = str(config.get("ENV_NAME", "Seaquest"))
    _configure_ego_obs_format(config)

    manager = EnvManager()
    components = manager.get_components(env_name)
    env, _ = manager.make_env(env_name, config, train=True)
    test_env, renderer = manager.make_env(env_name, config, train=False)

    heuristic_mode = config.get("HEURISTIC_MODE", "env_specific")
    if heuristic_mode == "nearest":
        no_target_idx = components.no_target_idx
        heuristic_fn = functools.partial(
            get_nearest_object_target, no_target_idx=no_target_idx
        )
    else:
        heuristic_fn = components.get_heuristic_target

    train_fn = make_tandem_train(
        config,
        env,
        test_env,
        renderer,
        no_target_idx=components.no_target_idx,
        get_heuristic_target=heuristic_fn,
        action_mapper=components.action_mapper,
        inverse_action_mapper=components.inverse_action_mapper,
        extract_all_skills_rewards=components.reward_extractor,
        extract_life_lost=components.extract_life_lost,
        meta_mask_hook=components.meta_mask_hook,
        forced_target_hook=components.forced_target_hook,
    )
    return train_fn, renderer


def single_run(config):
    """
    Executes a single training run with Weights & Biases logging.
    """
    # Flatten: experiment yaml -> ``experiment`` group; alg block merged on top.
    exp = config.get("experiment", {})
    root = {k: v for k, v in config.items() if k != "experiment"}
    base_config = {
        **root,
        **{k: v for k, v in exp.items() if k != "alg"},
        **exp.get("alg", {}),
    }

    alg_name = base_config.get("ALG_NAME", "hierarchical_pqn")
    env_name = str(base_config.get("ENV_NAME", "Seaquest"))
    _configure_ego_obs_format(base_config)

    if base_config.get("WANDB_MODE", "disabled") != "disabled":
        wandb.init(
            entity=base_config.get("ENTITY", None),
            project=base_config.get("PROJECT", "concept_based_learning"),
            tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"]
                 + list(base_config.get("WANDB_TAGS", [])),
            name=base_config.get("WANDB_RUN_NAME", f'{alg_name}_{env_name}'),
            config=base_config,
            mode=base_config["WANDB_MODE"],
        )

    rng = jax.random.PRNGKey(base_config["SEED"])
    eval_seed = base_config.get("EVAL_SEED", base_config["SEED"] + 1)
    num_seeds = base_config.get("NUM_SEEDS", 1)

    global_t0 = time.time()
    train_mode = str(base_config.get("TRAIN_MODE", "skills_only"))
    print(f"\n=== Starting Training Run in Mode: {train_mode.upper()} ===")

    if start_rtpt(base_config):
        print("RTPT progress tracking enabled.")

    # Optional: initialize skill network weights from a pre-trained model
    # to enable few-shot / transfer learning across environments.
    init_model_path = base_config.get("INIT_MODEL_PATH", None)
    prev_params = None
    if init_model_path is not None:
        _configure_ego_obs_format(base_config)
        print(f"Initializing skills from pre-trained model: {init_model_path}")
        prev_params = load_params_from_file(init_model_path)
        inferred = infer_skill_hparams_from_params(prev_params)
        ckpt_ego_in = int(inferred.get("_ckpt_ego_in", -1))
        expected_ego_in = int(ego_obs_channels_per_frame()) * int(
            base_config.get("FRAME_STACK_SIZE", 4)
        )
        if ckpt_ego_in != expected_ego_in:
            warnings.warn(
                "Checkpoint ego feature length mismatch: "
                f"checkpoint expects {ckpt_ego_in}, current config expects "
                f"{expected_ego_in} (channels/frame={ego_obs_channels_per_frame()}, "
                f"FRAME_STACK_SIZE={base_config.get('FRAME_STACK_SIZE', 4)}). "
                "This commonly happens when USE_EGO_SIZE_FEATURES / "
                "USE_EGO_VELOCITY_FEATURES / USE_LEGACY_EGO_OBS differ between "
                "checkpoint training and current run.",
                UserWarning,
                stacklevel=2,
            )

    def run_skill_curriculum(params_to_start_with, steps_to_run):
        """Runs System 1 training with optional curriculum phases."""
        nonlocal rng
        current_params = params_to_start_with
        manager = EnvManager()
        components = manager.get_components(base_config.get("ENV_NAME", "Seaquest"))
        env_key = str(base_config.get("ENV_NAME", "")).lower()

        phases = build_skill_phases(
            base_config=base_config,
            total_timesteps=int(steps_to_run),
            env_key=env_key,
            resolve_curriculum_mods=components.resolve_curriculum_mods,
        )
        n_phases = len(phases)

        for phase_idx, phase in enumerate(phases):
            print(f"\n=== Skill Phase {phase_idx + 1}/{n_phases}: {phase.name} ===")

            phase_config = copy.deepcopy(base_config)
            phase_config["TOTAL_TIMESTEPS"] = int(phase.total_timesteps)
            if phase.game_mods:
                phase_config["GAME_MODS"] = list(phase.game_mods)
            else:
                phase_config.pop("GAME_MODS", None)

            num_updates_this_phase = (
                phase_config["TOTAL_TIMESTEPS"]
                // phase_config.get("NUM_STEPS", 128)
                // phase_config.get("NUM_ENVS", 512)
            )
            phase_divisor = int(phase_config.get("NUM_STEPS", 128)) * int(
                phase_config.get("NUM_ENVS", 512)
            )
            dropped = int(phase_config["TOTAL_TIMESTEPS"]) % max(1, phase_divisor)
            if dropped != 0:
                warnings.warn(
                    f"TOTAL_TIMESTEPS={phase_config['TOTAL_TIMESTEPS']} is not divisible by "
                    f"NUM_STEPS*NUM_ENVS={phase_divisor}; dropping {dropped} timesteps in "
                    f"phase '{phase.name}'.",
                    UserWarning,
                    stacklevel=2,
                )
            phase_config["STEP_OFFSET"] = int(phase.step_offset_updates)

            rng, rng_phase = jax.random.split(rng)
            rngs = jax.random.split(rng_phase, num_seeds)

            print(f"Compiling skill trainer for {env_name} (phase: {phase.name})...")
            start_compile = time.time()
            train_fn, eval_rollout_fn, renderer = outer_make_train(phase_config)
            train_vjit = jax.jit(jax.vmap(train_fn, in_axes=(0, None, None)))
            train_vjit.lower(rngs, current_params, None).compile()
            print(
                f"Skill compilation for phase '{phase.name}' "
                f"took {time.time() - start_compile:.2f} seconds."
            )

            phase_t0 = time.time()
            outs = jax.block_until_ready(train_vjit(rngs, current_params, None))
            print(
                f"Skill phase '{phase.name}' completed "
                f"in {time.time() - phase_t0:.2f} seconds."
            )

            runner_state = outs["runner_state"]
            train_states, *_ = runner_state
            first_train_states = jax.tree_util.tree_map(lambda x: x[0], train_states)
            current_params = first_train_states.params

            global_step_end_of_phase = int(phase.step_offset_updates) + int(num_updates_this_phase)
            if (
                phase_config.get("RECORD_VIDEO")
                and phase_config.get("VIDEO_AT_END", True)
                and eval_rollout_fn is not None
            ):
                video_key = f"video/{phase_idx + 1:02d}_{phase.name}"
                video_output_path = None
                if (
                    phase_config.get("WANDB_MODE", "disabled") == "disabled"
                    or wandb.run is None
                ):
                    save_root = resolve_config_path(
                        phase_config.get("SAVE_PATH", "./outputs/models")
                    )
                    vdir = os.path.join(
                        save_root,
                        "curriculum_videos",
                        (
                            f"{phase_config.get('ALG_NAME', 'run')}_"
                            f"{phase_config.get('ENV_NAME', 'env')}_"
                            f"seed{phase_config.get('SEED', 0)}"
                        ),
                    )
                    os.makedirs(vdir, exist_ok=True)
                    safe_key = video_key.replace("/", "_")
                    video_output_path = os.path.join(
                        vdir,
                        f"{safe_key}_step{global_step_end_of_phase}.mp4",
                    )

                use_wandb_video = (
                    phase_config.get("WANDB_MODE", "disabled") != "disabled"
                    and wandb.run is not None
                )
                if use_wandb_video or video_output_path:
                    print(f"Recording end-of-phase video for '{phase.name}'...")
                    eval_rng = jax.random.PRNGKey(eval_seed)
                    (
                        states,
                        active_agents,
                        combined_qs,
                        dones,
                        target_obj_idxs,
                        objects_bbox,
                    ) = eval_rollout_fn(
                        runner_state,
                        eval_rng,
                        global_step_end_of_phase,
                        phase_config.get("VIDEO_MAX_FRAMES", 2000),
                    )
                    video_callback(
                        states,
                        active_agents,
                        combined_qs,
                        dones,
                        global_step_end_of_phase,
                        renderer,
                        target_obj_idxs=target_obj_idxs,
                        objects_bbox=objects_bbox,
                        video_key=video_key,
                        wandb_step=global_step_end_of_phase,
                        video_output_path=(
                            video_output_path if not use_wandb_video else None
                        ),
                    )
                    wait_for_video_finish()
                else:
                    print(
                        f"RECORD_VIDEO/VIDEO_AT_END set for phase '{phase.name}' "
                        "but wandb is unavailable and SAVE_PATH could not be used; "
                        "skipping video."
                    )
        last_global_step = int(phases[-1].step_offset_updates) + int(
            base_config.get("TOTAL_TIMESTEPS", 0) // base_config.get("NUM_STEPS", 128) // base_config.get("NUM_ENVS", 512)
        )
        # More robust: return the computed end step from the last phase
        last_global_step = global_step_end_of_phase if phases else 0
        return current_params, last_global_step

    def run_meta_training(skill_params, steps_to_run, step_offset=0, meta_params_to_start=None):
        """Runs System 2 training strictly on the unmodified base environment."""
        nonlocal rng
        if skill_params is None:
            raise ValueError("Meta training requires skill parameters.")

        print("\n=== Meta-Policy Training (Base Environment) ===")
        meta_config = copy.deepcopy(base_config)
        meta_config["TOTAL_TIMESTEPS"] = int(steps_to_run)
        meta_config["STEP_OFFSET"] = int(step_offset)
        meta_config.pop("GAME_MODS", None)
        meta_config.pop("EVAL_GAME_MODS", None)

        num_updates = (
            meta_config["TOTAL_TIMESTEPS"]
            // meta_config.get("NUM_STEPS", 128)
            // meta_config.get("NUM_ENVS", 512)
        )

        rng, rng_meta = jax.random.split(rng)
        rngs = jax.random.split(rng_meta, num_seeds)

        start_compile = time.time()
        train_fn, eval_rollout_fn, renderer = outer_make_meta_train(
            meta_config, frozen_skill_params=skill_params
        )
        train_vjit = jax.jit(jax.vmap(train_fn, in_axes=(0, None)))
        train_vjit.lower(rngs, meta_params_to_start).compile()
        print(f"Meta compilation took {time.time() - start_compile:.2f} seconds.")

        meta_t0 = time.time()
        outs = jax.block_until_ready(train_vjit(rngs, meta_params_to_start))
        print(f"Meta training completed in {time.time() - meta_t0:.2f} seconds.")

        runner_state = outs["runner_state"]
        train_state, *_ = runner_state
        first_train_state = jax.tree_util.tree_map(lambda x: x[0], train_state)
        meta_params = first_train_state.params

        global_step_end = step_offset + num_updates
        if (
            meta_config.get("RECORD_VIDEO")
            and meta_config.get("VIDEO_AT_END", True)
            and eval_rollout_fn is not None
        ):
            video_output_path = None
            if (
                meta_config.get("WANDB_MODE", "disabled") == "disabled"
                or wandb.run is None
            ):
                save_root = resolve_config_path(
                    meta_config.get("SAVE_PATH", "./outputs/models")
                )
                vdir = os.path.join(
                    save_root,
                    "curriculum_videos",
                    (
                        f"{meta_config.get('ALG_NAME', 'run')}_"
                        f"{meta_config.get('ENV_NAME', 'env')}_"
                        f"seed{meta_config.get('SEED', 0)}"
                    ),
                )
                os.makedirs(vdir, exist_ok=True)
                video_output_path = os.path.join(
                    vdir,
                    f"video_meta_policy_step{global_step_end}.mp4",
                )

            use_wandb_video = (
                meta_config.get("WANDB_MODE", "disabled") != "disabled"
                and wandb.run is not None
            )
            if use_wandb_video or video_output_path:
                print("Recording end-of-meta-training video...")
                eval_rng = jax.random.PRNGKey(eval_seed)
                (
                    states,
                    active_agents,
                    combined_qs,
                    dones,
                    target_obj_idxs,
                    objects_bbox,
                ) = eval_rollout_fn(
                    runner_state,
                    eval_rng,
                    global_step_end,
                    meta_config.get("VIDEO_MAX_FRAMES", 2000),
                )
                video_callback(
                    states,
                    active_agents,
                    combined_qs,
                    dones,
                    global_step_end,
                    renderer,
                    target_obj_idxs=target_obj_idxs,
                    objects_bbox=objects_bbox,
                    video_key="video/meta_policy",
                    wandb_step=global_step_end,
                    video_output_path=(
                        video_output_path if not use_wandb_video else None
                    ),
                )
                wait_for_video_finish()
            else:
                print(
                    "RECORD_VIDEO/VIDEO_AT_END set for meta training "
                    "but wandb is unavailable and SAVE_PATH could not be used; "
                    "skipping video."
                )

        return meta_params, global_step_end, skill_params

    def run_tandem_training(steps_to_run, init_meta_params=None, init_skill_params=None):
        """Runs joint System 1 + System 2 training in a single optimization loop."""
        nonlocal rng
        tandem_config = copy.deepcopy(base_config)
        tandem_config["TOTAL_TIMESTEPS"] = int(steps_to_run)

        tandem_divisor = int(tandem_config.get("NUM_STEPS", 128)) * int(
            tandem_config.get("NUM_ENVS", 512)
        )
        dropped = int(steps_to_run) % max(1, tandem_divisor)
        if dropped != 0:
            warnings.warn(
                f"TOTAL_TIMESTEPS={steps_to_run} is not divisible by NUM_STEPS*NUM_ENVS="
                f"{tandem_divisor}; dropping {dropped} timesteps in tandem training.",
                UserWarning,
                stacklevel=2,
            )

        rng, rng_tandem = jax.random.split(rng)
        rngs = jax.random.split(rng_tandem, num_seeds)

        print("\n=== Tandem Training (Joint Skills + Meta-Policy) ===")
        start_compile = time.time()
        train_fn, _ = outer_make_tandem_train(tandem_config)
        train_vjit = jax.jit(jax.vmap(train_fn, in_axes=(0, None, None)))
        train_vjit.lower(rngs, init_meta_params, init_skill_params).compile()
        print(f"Tandem compilation took {time.time() - start_compile:.2f} seconds.")

        tandem_t0 = time.time()
        outs = jax.block_until_ready(train_vjit(rngs, init_meta_params, init_skill_params))
        print(f"Tandem training completed in {time.time() - tandem_t0:.2f} seconds.")

        meta_state, skill_states, *_ = outs["runner_state"]
        first_meta_state = jax.tree_util.tree_map(lambda x: x[0], meta_state)
        first_skill_states = jax.tree_util.tree_map(lambda x: x[0], skill_states)
        final_skill_params = first_skill_states.params
        final_meta_params = first_meta_state.params

        num_updates = (
            int(tandem_config["TOTAL_TIMESTEPS"])
            // int(tandem_config.get("NUM_STEPS", 128))
            // int(tandem_config.get("NUM_ENVS", 512))
        )
        global_step_end = num_updates
        if tandem_config.get("RECORD_VIDEO") and tandem_config.get("VIDEO_AT_END", True):
            video_output_path = None
            if (
                tandem_config.get("WANDB_MODE", "disabled") == "disabled"
                or wandb.run is None
            ):
                save_root = resolve_config_path(
                    tandem_config.get("SAVE_PATH", "./outputs/models")
                )
                vdir = os.path.join(
                    save_root,
                    "curriculum_videos",
                    (
                        f"{tandem_config.get('ALG_NAME', 'run')}_"
                        f"{tandem_config.get('ENV_NAME', 'env')}_"
                        f"seed{tandem_config.get('SEED', 0)}"
                    ),
                )
                os.makedirs(vdir, exist_ok=True)
                video_output_path = os.path.join(
                    vdir,
                    f"video_tandem_step{global_step_end}.mp4",
                )

            use_wandb_video = (
                tandem_config.get("WANDB_MODE", "disabled") != "disabled"
                and wandb.run is not None
            )
            if use_wandb_video or video_output_path:
                print("Recording end-of-tandem-training video...")
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
                    tandem_config,
                    params_tree=final_skill_params,
                    meta_params=final_meta_params,
                    max_frames=tandem_config.get("VIDEO_MAX_FRAMES", 2000),
                )
                video_callback(
                    states,
                    active_agents,
                    combined_qs,
                    dones,
                    global_step_end,
                    renderer,
                    target_obj_idxs=target_obj_idxs,
                    objects_bbox=objects_bbox,
                    distances=distances,
                    angles=angles,
                    video_key="video/tandem",
                    wandb_step=global_step_end,
                    video_output_path=(
                        video_output_path if not use_wandb_video else None
                    ),
                )
                wait_for_video_finish()
            else:
                print(
                    "RECORD_VIDEO/VIDEO_AT_END set for tandem training "
                    "but wandb is unavailable and SAVE_PATH could not be used; "
                    "skipping video."
                )

        return final_skill_params, final_meta_params

    final_skill_params = None
    final_meta_params = None

    total_timesteps = int(base_config.get("TOTAL_TIMESTEPS", 0))
    if train_mode == "skills_only":
        final_skill_params, _ = run_skill_curriculum(prev_params, total_timesteps)
    elif train_mode == "meta_only":
        if prev_params is None:
            raise ValueError(
                "Mode 'meta_only' requires INIT_MODEL_PATH containing skill parameters."
            )
        final_skill_params = prev_params
        meta_divisor = int(base_config.get("NUM_STEPS", 128)) * int(
            base_config.get("NUM_ENVS", 512)
        )
        dropped = int(total_timesteps) % max(1, meta_divisor)
        if dropped != 0:
            warnings.warn(
                f"TOTAL_TIMESTEPS={total_timesteps} is not divisible by NUM_STEPS*NUM_ENVS="
                f"{meta_divisor}; dropping {dropped} timesteps in meta training.",
                UserWarning,
                stacklevel=2,
            )
        final_meta_params, _, final_skill_params = run_meta_training(
            final_skill_params, total_timesteps, step_offset=0
        )
    elif train_mode == "tandem":
        final_skill_params, final_meta_params = run_tandem_training(
            total_timesteps,
            init_meta_params=None,
            init_skill_params=prev_params,
        )
    else:
        raise ValueError(
            f"Unknown TRAIN_MODE: {train_mode}. Supported modes: skills_only, meta_only, tandem."
        )

    print(f"\nTraining run completed in {time.time() - global_t0:.2f} seconds.")

    # Optional: Save final config (hyperparameters) and model parameters to disk
    save_path = resolve_config_path(base_config.get("SAVE_PATH", None))
    if save_path is not None:
        save_dir = os.path.join(save_path, env_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save config
        OmegaConf.save(
            base_config,
            os.path.join(save_dir, f'{alg_name}_{env_name}_seed{base_config["SEED"]}_config.yaml'),
        )
        print(f"Config saved to {save_dir}")

        if final_skill_params is not None:
            flat_skill_params = flatten_dict(final_skill_params, sep=",")
            if final_meta_params is None:
                skills_filename = f'{alg_name}_{env_name}_seed{base_config["SEED"]}.safetensors'
            else:
                skills_filename = (
                    f'{alg_name}_{env_name}_seed{base_config["SEED"]}_skills.safetensors'
                )
            skills_model_path = _safe_checkpoint_path(
                os.path.join(save_dir, skills_filename),
                init_model_path=init_model_path,
            )
            save_file(flat_skill_params, skills_model_path)
            print(f"Skill model parameters saved to {skills_model_path}")

        if final_meta_params is not None:
            flat_meta_params = flatten_dict(final_meta_params, sep=",")
            meta_model_path = _safe_checkpoint_path(
                os.path.join(
                    save_dir,
                    f'{alg_name}_{env_name}_seed{base_config["SEED"]}_meta.safetensors',
                ),
                init_model_path=init_model_path,
            )
            save_file(flat_meta_params, meta_model_path)
            print(f"Meta-policy parameters saved to {meta_model_path}")

    if wandb.run is not None:
        wandb.finish()


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config):
    # Convert OmegaConf to standard dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)
    print("Initializing training with configuration:\n", OmegaConf.to_yaml(config_dict))
    
    single_run(config_dict)


if __name__ == "__main__":
    main()