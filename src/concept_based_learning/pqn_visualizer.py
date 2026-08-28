import argparse
import os
from pathlib import Path

# Pin a real SDL video driver *before* importing pygame. If this is left unset
# (or set to "dummy"), pygame can initialize a headless surface and never show
# a window — which looks like "pygame is running but nothing spawns".
_sdl_driver = str(os.environ.get("SDL_VIDEODRIVER", "")).lower().strip()
if _sdl_driver in ("", "dummy", "auto"):
    os.environ["SDL_VIDEODRIVER"] = "x11"

import jax
import jax.numpy as jnp
import numpy as np
import pygame
import wandb
from omegaconf import OmegaConf
from safetensors.flax import load_file
from flax.traverse_util import unflatten_dict

from concept_based_learning.observations.action_space import (
    fire_ego_action_indices,
    make_realized_ego_mask_fn,
    mask_q_with_legal_ego,
)
from concept_based_learning.observations.egocentric import (
    MAX_DIST,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    aabb_surface_gap_px,
    ego_distance_bins,
    ego_distance_flag_thresholds_px,
    ego_obs_channels_per_frame,
    ego_pose_channels_per_frame,
    _get_player,
    _center_y,
    _uses_bottom_edge_y,
    configure_ego_obs_from_config,
    distance_pixels_from_feature,
    get_ego_obs,
    slice_skill_ego,
    using_legacy_ego_obs,
)
from concept_based_learning.training.skill_params import (
    as_per_skill_param_list,
    dense0_in_dim,
)
from concept_based_learning.environments.datatypes import (
    ConceptObservation,
    EgocentricAction,
    LABEL_AVOID,
    configure_label_routing_from_config,
)
from concept_based_learning.environments.env_manager import EnvManager
from concept_based_learning.environments.heuristics import (
    initial_policy_state,
    resolve_heuristic_target_fn,
    resolve_heuristic_with_mission_fn,
    resolve_heuristic_with_state_fn,
)
from concept_based_learning.environments.choppercommand.choppercommand_policy import (
    debug_heuristic_snapshot,
)
from concept_based_learning.networks.meta_factory import create_meta_network
from concept_based_learning.networks.meta_network import flatten_raw_observation
from concept_based_learning.paper_eval.env_metrics import (
    aggregate_episode_metrics,
    init_episode_metrics,
    make_metric_fns,
    metrics_to_python,
)
from concept_based_learning.environments.meta_target_mask import (
    build_meta_target_mask,
    is_meta_fire_target,
    is_meta_noop_target,
    meta_fire_game_action,
    meta_fire_target_enabled,
    meta_fire_target_idx,
    meta_noop_target_enabled,
    meta_noop_target_idx,
)
from concept_based_learning.environments.meta_target_persistence import (
    commit_meta_targets,
    compute_meta_target_locked,
    meta_q_stickiness_enabled,
    meta_target_persist_enabled,
)
from concept_based_learning.networks.skill_network import (
    PIXEL_SKILL_INPUT_MODES,
    QNetwork,
    q_network_from_config,
)
from concept_based_learning.observations.pov_relative import resolve_action_mappers
from concept_based_learning.training.episode_stats import unwrap_base_env_state
from concept_based_learning.video_recorder import (
    video_callback,
    wait_for_video_finish,
    draw_target_outline,
)


def _project_root() -> Path:
    """Return the repository root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


def apply_visualizer_config_overrides(
    config: dict,
    *,
    no_outline: bool = False,
    black_pixel: bool = False,
    heuristic_mode: str | None = None,
) -> dict:
    """Apply CLI ablation flags for live/recorded visualizer runs."""
    if no_outline:
        config["SKILL_PIXEL_PAINT_TARGET_OUTLINE"] = False
        config["VISUALIZER_DRAW_TARGET_OUTLINE"] = False
    if black_pixel:
        config["SKILL_PIXEL_BLACKOUT"] = True
    if heuristic_mode is not None:
        config["HEURISTIC_MODE"] = str(heuristic_mode)
    return config


def _prepare_pygame_env() -> None:
    """Strip IME hooks and pin a real SDL video driver before display init."""
    for key in ("SDL_IM_MODULE", "XMODIFIERS", "GTK_IM_MODULE", "QT_IM_MODULE"):
        os.environ.pop(key, None)
    # "dummy" creates no window; force a real driver unless the user set one.
    current = str(os.environ.get("SDL_VIDEODRIVER", "")).lower().strip()
    if current in ("", "dummy", "auto"):
        if (
            os.environ.get("WAYLAND_DISPLAY")
            and str(os.environ.get("XDG_SESSION_TYPE", "")).lower() == "wayland"
            and not os.environ.get("DISPLAY")
        ):
            os.environ["SDL_VIDEODRIVER"] = "wayland"
        else:
            os.environ["SDL_VIDEODRIVER"] = "x11"
    # Prefer a centered window on multi-monitor setups (left-edge is easy to miss).
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")


def _raise_pygame_window() -> None:
    """Best-effort focus/raise so the SDL window is not buried under other apps."""
    import subprocess

    try:
        wm = pygame.display.get_wm_info()
    except pygame.error:
        return
    wid = wm.get("window")
    if not wid:
        return
    hex_id = hex(int(wid))
    for argv in (
        ["wmctrl", "-i", "-a", hex_id],
        ["xdotool", "windowactivate", "--sync", str(int(wid))],
        ["xdotool", "windowraise", str(int(wid))],
    ):
        try:
            subprocess.run(
                argv,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.0,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            continue


def bootstrap_pygame_window(
    window_name: str = "Hierarchical PQN Live Visualization",
    window_size: tuple[int, int] = (480, 630),
) -> tuple:
    """
    Create the pygame window *before* heavy JAX/GPU work / OpenCV import.

    Returns ``(screen, font)``. Calling this early avoids SDL picking the
    ``dummy`` driver (no window) once JAX or OpenCV have touched the process.
    """
    _prepare_pygame_env()
    # If a previous init selected the wrong driver, tear it down and retry.
    if pygame.get_init() or pygame.display.get_init():
        try:
            if pygame.display.get_init() and pygame.display.get_driver() == "dummy":
                pygame.display.quit()
                pygame.quit()
        except pygame.error:
            pygame.quit()

    if not pygame.get_init():
        pygame.init()
    elif not pygame.display.get_init():
        pygame.display.init()

    pygame.display.set_caption(window_name)
    flags = pygame.SHOWN | pygame.RESIZABLE
    screen = pygame.display.set_mode(window_size, flags)
    font = pygame.font.Font(None, 28)
    screen.fill((20, 20, 20))
    splash = font.render("Starting visualizer...", True, (240, 240, 240))
    screen.blit(splash, (16, window_size[1] // 2 - 10))
    pygame.display.flip()
    pygame.event.pump()
    _raise_pygame_window()

    driver = pygame.display.get_driver()
    active = pygame.display.get_active()
    print(
        f"[live-viz] pygame window opened "
        f"(driver={driver}, size={screen.get_size()}, active={active}, "
        f"DISPLAY={os.environ.get('DISPLAY')}, "
        f"SDL_VIDEODRIVER={os.environ.get('SDL_VIDEODRIVER')})",
        flush=True,
    )
    if driver == "dummy":
        raise RuntimeError(
            "pygame selected the SDL 'dummy' video driver, so no window will appear. "
            "Unset SDL_VIDEODRIVER (or set it to x11), ensure DISPLAY is set, "
            "or re-run with --display opencv."
        )
    if not active:
        print(
            "[live-viz] warning: pygame window reports inactive — "
            "it may be on another monitor/workspace; try Alt-Tab",
            flush=True,
        )
    return screen, font


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
    known = {
        "seaquest",
        "phoenix",
        "choppercommand",
        "spaceinvaders",
        "montezumarevenge",
        "montezuma_revenge",
        "venture",
        "asterix",
        "beamrider",
        "kangaroo",
        "freeway",
        "pong",
        "tennis",
        "breakout",
        "skiing",
    }
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


def _destroy_skill_idx(config: dict) -> int:
    names = [str(n) for n in (config.get("SKILL_NAMES") or [])]
    return names.index("DESTROY") if "DESTROY" in names else 0


def _unwrap_stacked_skill_params(params_tree):
    """Return a single-skill param subtree and the number of stacked skills."""
    packed = as_per_skill_param_list(params_tree)
    if packed is not None:
        return packed[0], len(packed)
    kernel = params_tree["Dense_0"]["kernel"]
    arr = np.asarray(kernel)
    if arr.ndim == 3:
        num_skills = int(arr.shape[0])
        single = jax.tree_util.tree_map(lambda x: x[0], params_tree)
        return single, num_skills
    return params_tree, 1


def infer_skill_hparams_from_params(
    params_tree,
    *,
    skill_input_mode: str = "ego",
) -> dict:
    """
    Infer skill Q-network hyperparameters from a checkpoint parameter tree.

    Expects Flax ``QNetwork`` param keys ``Dense_0`` … ``Dense_{NUM_LAYERS}``.

    ``Dense_0`` input size is egocentric length only for ``ego`` / ``ego_embed``.
    For ``oc_embed`` / ``pixel_embed`` it is the embedding width — do **not**
    treat it as ``channels × FRAME_STACK_SIZE``.
    """
    sample, num_skills = _unwrap_stacked_skill_params(params_tree)
    packed = as_per_skill_param_list(params_tree)
    dense_indices = sorted(
        int(key.split("_", 1)[1])
        for key in sample
        if key.startswith("Dense_") and key.split("_", 1)[1].isdigit()
    )
    if not dense_indices:
        raise ValueError("Skill checkpoint has no Dense_* layers; cannot infer architecture.")

    first = np.asarray(sample[f"Dense_{dense_indices[0]}"]["kernel"])
    last = np.asarray(sample[f"Dense_{dense_indices[-1]}"]["kernel"])
    if packed is not None:
        in_dims = [dense0_in_dim(p) for p in packed]
        first_in = max(in_dims)
    else:
        in_dims = None
        first_in = int(first.shape[0])
    hidden_size = int(first.shape[-1])
    ego_action_dim = int(last.shape[-1])
    num_layers = len(dense_indices) - 1

    mode = str(skill_input_mode or "ego").lower().strip()
    frame_stack_size = None
    ckpt_ego_in = None
    if mode in ("ego", "ego_embed"):
        channels_per_frame = ego_obs_channels_per_frame()
        if channels_per_frame > 0 and first_in % channels_per_frame == 0:
            frame_stack_size = first_in // channels_per_frame
        elif ego_pose_channels_per_frame() > 0:
            pose_in = min(in_dims) if in_dims else first_in
            pose_ch = ego_pose_channels_per_frame()
            if pose_in % pose_ch == 0:
                frame_stack_size = pose_in // pose_ch
        ckpt_ego_in = first_in

    out = {
        "HIDDEN_SIZE": hidden_size,
        "NUM_LAYERS": num_layers,
        "EGO_ACTION_DIM": ego_action_dim,
        "NUM_SKILLS": num_skills,
        "FRAME_STACK_SIZE": frame_stack_size,
        "_ckpt_ego_in": ckpt_ego_in,
        "_ckpt_dense0_in": first_in,
    }
    if in_dims is not None:
        out["_ckpt_skill_in_dims"] = in_dims
    return out


def ensure_skill_params_stack(params_tree, num_skills: int):
    """Ensure skill params are stacked along axis 0 for ``jax.vmap`` over skills.

    Packed per-skill checkpoints (different ``Dense_0`` widths) stay a tuple.
    """
    packed = as_per_skill_param_list(params_tree)
    if packed is not None:
        if len(packed) != int(num_skills):
            raise ValueError(
                f"Packed skill checkpoint has {len(packed)} heads, config NUM_SKILLS={num_skills}"
            )
        return tuple(packed)
    kernel = np.asarray(params_tree["Dense_0"]["kernel"])
    if kernel.ndim == 2:
        return jax.tree_util.tree_map(
            lambda p: jnp.stack([p] * num_skills, axis=0),
            params_tree,
        )
    return params_tree


def apply_skill_q_all(
    skill_network,
    params_tree,
    ego_full,
    *,
    target_idx,
    pixel_obs=None,
    objects=None,
    stack_size: int,
    destroy_skill_idx: int = 0,
):
    """Apply every skill head. Slices pose-only ego for non-DESTROY when packed."""
    packed = as_per_skill_param_list(params_tree)
    if packed is None:

        def apply_one(params_for_skill):
            return skill_network.apply(
                {"params": params_for_skill},
                ego_full,
                train=False,
                target_idx=target_idx,
                pixel_obs=pixel_obs,
                objects=objects,
            )

        return jax.vmap(apply_one)(params_tree)

    qs = []
    for i, params_for_skill in enumerate(packed):
        ego_i = slice_skill_ego(
            ego_full,
            i,
            stack_size=int(stack_size),
            destroy_skill_idx=int(destroy_skill_idx),
        )
        qs.append(
            skill_network.apply(
                {"params": params_for_skill},
                ego_i,
                train=False,
                target_idx=target_idx,
                pixel_obs=pixel_obs,
                objects=objects,
            )
        )
    return jnp.stack(qs)


def align_config_with_skill_checkpoint(config: dict, params_tree) -> dict:
    """Override skill architecture fields so the network matches the checkpoint."""
    # Sync ego channel globals before inferring stack size from Dense_0.
    configure_ego_obs_from_config(config)
    configure_label_routing_from_config(config)
    mode = str(config.get("SKILL_INPUT_MODE", "ego")).lower().strip()
    inferred = infer_skill_hparams_from_params(params_tree, skill_input_mode=mode)
    ckpt_ego_in = inferred.pop("_ckpt_ego_in", None)
    inferred.pop("_ckpt_dense0_in", None)
    inferred.pop("_ckpt_skill_in_dims", None)
    frame_stack_size = inferred.get("FRAME_STACK_SIZE")
    # Never clobber FRAME_STACK_SIZE with a bogus embed_dim / channel quotient.
    updated = {
        **config,
        **{
            k: v
            for k, v in inferred.items()
            if v is not None and k != "FRAME_STACK_SIZE"
        },
    }
    if frame_stack_size is not None:
        prev = int(config.get("FRAME_STACK_SIZE", frame_stack_size))
        if prev != int(frame_stack_size):
            print(
                f"Using checkpoint FRAME_STACK_SIZE={frame_stack_size} "
                f"(default config had {config.get('FRAME_STACK_SIZE')})."
            )
        updated["FRAME_STACK_SIZE"] = int(frame_stack_size)
    if int(config.get("HIDDEN_SIZE", -1)) != int(updated["HIDDEN_SIZE"]):
        print(
            f"Using checkpoint HIDDEN_SIZE={updated['HIDDEN_SIZE']} "
            f"(default config had {config.get('HIDDEN_SIZE')})."
        )
    if ckpt_ego_in is not None:
        updated["_ckpt_ego_in"] = ckpt_ego_in
    return updated


def validate_skill_checkpoint_ego_dim(config: dict, ego_feat_dim: int) -> None:
    """Fail fast when the eval env's ego observation length does not match the checkpoint."""
    mode = str(config.get("SKILL_INPUT_MODE", "ego")).lower().strip()
    ckpt_in = config.pop("_ckpt_ego_in", None)
    # oc/pixel Dense_0 is embed width — not comparable to ego feat length.
    if mode not in ("ego", "ego_embed"):
        return
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

    Paper / greedy metrics always use **unclipped** env rewards so hierarchical
    skill evals are comparable to flat PQN and to ``avg_game_return``.
    """
    env_name = config.get("ENV_NAME", "Seaquest")
    manager = EnvManager()
    eval_cfg = dict(config)
    eval_cfg["CLIP_REWARD"] = False
    env, renderer = manager.make_env(env_name, eval_cfg, train=False)
    return env, renderer


def label_to_skill_idx(label, num_skills: int):
    """Delegate to the shared mapping in ``environments.datatypes``."""
    from concept_based_learning.environments.datatypes import (
        label_to_skill_idx as _shared,
    )

    return _shared(label, num_skills)


def _hud_target_distance_px(raw_obs, objects, target_obj_idx: int, last_frame_ego) -> float:
    """True AABB surface-gap px for overlays (bins/flags make ego[0] non-invertible)."""
    if ego_distance_bins() is not None or ego_distance_flag_thresholds_px():
        player = _get_player(raw_obs)
        y_bottom = _uses_bottom_edge_y(raw_obs)
        return float(
            aabb_surface_gap_px(
                player.x[-1],
                player.y[-1],
                player.width[-1],
                player.height[-1],
                objects.x[-1, target_obj_idx],
                objects.y[-1, target_obj_idx],
                objects.width[-1, target_obj_idx],
                objects.height[-1, target_obj_idx],
                y_is_bottom_edge=y_bottom,
            )
        )
    return distance_pixels_from_feature(float(last_frame_ego[0]))


def _skill_input_from_obs(config: dict, raw_obs, objects, target_idx):
    mode = str(config.get("SKILL_INPUT_MODE", "ego")).lower().strip()
    if mode in ("ego", "ego_embed"):
        return get_ego_obs(raw_obs, objects, target_idx)
    if mode == "oc_embed":
        return flatten_raw_observation(raw_obs)
    if mode in PIXEL_SKILL_INPUT_MODES:
        return get_ego_obs(raw_obs, objects, target_idx)
    raise ValueError(
        f"Unknown SKILL_INPUT_MODE={mode!r}. "
        "Expected one of: ego, ego_embed, oc_embed, pixel_embed, pixel_embed_cond."
    )


def _skill_pixel_obs_from_obs(config: dict, obs):
    """
    Pixel tensor for skill Q-network forward passes during eval/visualization.

    Only pixel modes read this input; ego/oc modes must pass None (not raw_observation,
    which is a structured game observation object).
    """
    mode = str(config.get("SKILL_INPUT_MODE", "ego")).lower().strip()
    if mode not in PIXEL_SKILL_INPUT_MODES:
        return None
    if obs.pixel_observation is None:
        raise ValueError(
            f"SKILL_INPUT_MODE={mode!r} requires pixel_observation from "
            "PixelAndObjectObsWrapper, but it is missing."
        )
    return obs.pixel_observation


def _propose_meta_target(
    obs,
    config,
    meta_network,
    meta_params,
    meta_mask_hook,
    forced_target_hook,
    no_target_idx,
    *,
    last_target_idx=None,
    suspended_mission_idx=None,
    persist_steps_remaining=None,
):
    """Greedy meta proposal before optional persistence override."""
    if last_target_idx is None:
        last_target_idx = no_target_idx
    if suspended_mission_idx is None:
        suspended_mission_idx = no_target_idx
    if persist_steps_remaining is None:
        persist_steps_remaining = jnp.array(0, dtype=jnp.int32)

    last_b = jnp.asarray([last_target_idx], dtype=jnp.int32)
    suspended_b = jnp.asarray([suspended_mission_idx], dtype=jnp.int32)
    steps_b = jnp.asarray([persist_steps_remaining], dtype=jnp.int32)
    target_locked = compute_meta_target_locked(
        last_b,
        suspended_b,
        obs.objects,
        config,
        no_target_idx=no_target_idx,
        max_objects=int(config["MAX_OBJECTS"]),
        persist_steps_remaining=steps_b,
    )[0]

    meta_q_vals = meta_network.apply(
        {"params": meta_params},
        obs.raw_observation,
        obs.objects,
        train=False,
        target_locked=target_locked,
        pixel_obs=obs.pixel_observation,
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
    proposed = jnp.where(any_active, target_idx, no_target_idx).astype(jnp.int32)
    return proposed, meta_q_vals, active_mask


def _select_target_with_meta(
    obs,
    config,
    meta_network,
    meta_params,
    meta_mask_hook,
    forced_target_hook,
    no_target_idx,
    *,
    last_target_idx=None,
    persist_steps_remaining=None,
    suspended_mission_idx=None,
):
    """Select a target index using a learned meta-policy checkpoint."""
    if last_target_idx is None:
        last_target_idx = no_target_idx
    if persist_steps_remaining is None:
        persist_steps_remaining = jnp.array(0, dtype=jnp.int32)
    if suspended_mission_idx is None:
        suspended_mission_idx = no_target_idx

    proposed, meta_q_vals, active_mask = _propose_meta_target(
        obs,
        config,
        meta_network,
        meta_params,
        meta_mask_hook,
        forced_target_hook,
        no_target_idx,
        last_target_idx=last_target_idx,
        suspended_mission_idx=suspended_mission_idx,
        persist_steps_remaining=persist_steps_remaining,
    )
    if not (meta_target_persist_enabled(config) or meta_q_stickiness_enabled(config)):
        return proposed, persist_steps_remaining, suspended_mission_idx

    last_b = jnp.asarray([last_target_idx], dtype=jnp.int32)
    proposed_b = jnp.asarray([proposed], dtype=jnp.int32)
    q_b = meta_q_vals[jnp.newaxis, ...] if meta_q_vals.ndim == 1 else meta_q_vals
    mask_b = active_mask[jnp.newaxis, ...] if active_mask.ndim == 1 else active_mask
    steps_b = jnp.asarray([persist_steps_remaining], dtype=jnp.int32)
    suspended_b = jnp.asarray([suspended_mission_idx], dtype=jnp.int32)

    committed, new_steps, new_suspended = commit_meta_targets(
        proposed_b,
        last_b,
        obs.objects,
        q_b,
        mask_b,
        config,
        no_target_idx=no_target_idx,
        max_objects=int(config["MAX_OBJECTS"]),
        persist_steps_remaining=steps_b,
        suspended_mission_idxs=suspended_b,
    )
    return committed[0], new_steps[0], new_suspended[0]


def _jax_select_target_with_meta(
    obs,
    config,
    meta_network,
    meta_params,
    meta_mask_hook,
    forced_target_hook,
    no_target_idx,
    *,
    last_target_idx,
    persist_steps_remaining,
    suspended_mission_idx,
):
    """JAX-traced target selection (no host sync), with optional persistence."""
    proposed, meta_q_vals, active_mask = _propose_meta_target(
        obs,
        config,
        meta_network,
        meta_params,
        meta_mask_hook,
        forced_target_hook,
        no_target_idx,
        last_target_idx=last_target_idx,
        suspended_mission_idx=suspended_mission_idx,
        persist_steps_remaining=persist_steps_remaining,
    )
    if not meta_target_persist_enabled(config):
        return proposed, persist_steps_remaining, suspended_mission_idx

    last_b = jnp.asarray([last_target_idx], dtype=jnp.int32)
    proposed_b = jnp.asarray([proposed], dtype=jnp.int32)
    q_b = meta_q_vals[jnp.newaxis, ...] if meta_q_vals.ndim == 1 else meta_q_vals
    mask_b = active_mask[jnp.newaxis, ...] if active_mask.ndim == 1 else active_mask
    steps_b = jnp.asarray([persist_steps_remaining], dtype=jnp.int32)
    suspended_b = jnp.asarray([suspended_mission_idx], dtype=jnp.int32)
    committed, new_steps, new_suspended = apply_meta_target_persistence(
        proposed_b,
        last_b,
        obs.objects,
        q_b,
        mask_b,
        config,
        no_target_idx=no_target_idx,
        max_objects=int(config["MAX_OBJECTS"]),
        persist_steps_remaining=steps_b,
        suspended_mission_idxs=suspended_b,
    )
    return committed[0], new_steps[0], new_suspended[0]


def _make_jitted_greedy_episode_runner(
    env,
    skill_network,
    skill_params,
    config,
    components,
    *,
    meta_network,
    meta_params,
    max_frames: int,
    num_skills: int,
    ego_action_dim: int,
    restrict_fire_for_non_destroy: bool,
):
    """Compile one full greedy episode (env + meta + skills) into a single JIT call."""
    env_name = config.get("ENV_NAME", "Seaquest")
    heuristic_target = resolve_heuristic_target_fn(config, env_name, components)
    heuristic_with_mission = resolve_heuristic_with_mission_fn(config, env_name)
    no_target_idx = int(components.no_target_idx)
    action_mapper, _ = resolve_action_mappers(components, config)
    forced_target_hook = components.forced_target_hook
    meta_mask_hook = components.meta_mask_hook

    noop_enabled = meta_noop_target_enabled(config)
    fire_enabled = meta_fire_target_enabled(config)
    noop_idx = jnp.int32(meta_noop_target_idx(config))
    fire_idx = jnp.int32(meta_fire_target_idx(config))
    fire_action = jnp.int32(meta_fire_game_action(config))
    no_target = jnp.int32(no_target_idx)
    fire_indices = fire_ego_action_indices(ego_action_dim)
    use_meta = meta_network is not None
    update_metrics, finalize_metrics = make_metric_fns(env_name, has_semantic_obs=True)
    use_legal_ego_mask = bool(config.get("EGO_LEGAL_ACTION_MASK", False))
    legal_ego_mask_fn = make_realized_ego_mask_fn(
        action_mapper,
        noop_action=int(components.ego_mapper_noop_action),
        fire_action=int(components.ego_mapper_fire_action),
        ego_action_dim=int(ego_action_dim),
    )

    def _skill_game_action(obs, target):
        is_noop = is_meta_noop_target(
            target, config, noop_idx=int(noop_idx), enabled=noop_enabled
        )
        is_fire = is_meta_fire_target(
            target, config, fire_idx=int(fire_idx), enabled=fire_enabled
        )
        idle_action = jnp.int32(int(config.get("NO_TARGET_GAME_ACTION", 0)))
        safe_target = jnp.where(target == no_target, 0, target)
        raw_label = obs.objects.semantic_label[-1, safe_target]
        sk_idx = label_to_skill_idx(raw_label, num_skills)
        ego_obs = _skill_input_from_obs(config, obs.raw_observation, obs.objects, safe_target)
        pixel_obs = _skill_pixel_obs_from_obs(config, obs)

        q_all = apply_skill_q_all(
            skill_network,
            skill_params,
            ego_obs,
            target_idx=safe_target,
            pixel_obs=pixel_obs,
            objects=obs.objects,
            stack_size=int(config.get("FRAME_STACK_SIZE", 4)),
            destroy_skill_idx=_destroy_skill_idx(config),
        )
        q_skill = q_all[sk_idx]
        if restrict_fire_for_non_destroy:
            q_skill = jax.lax.cond(
                sk_idx == 0,
                lambda q: q,
                lambda q: q.at[fire_indices].set(-jnp.inf),
                q_skill,
            )
        if use_legal_ego_mask:
            legal = legal_ego_mask_fn(obs.raw_observation, safe_target, obs.objects)
            q_skill = mask_q_with_legal_ego(q_skill, legal)
        ego_a = jnp.argmax(q_skill)
        game_a = action_mapper(ego_a, obs.raw_observation, target, obs.objects)
        game_a = jnp.where(is_fire, fire_action, game_a)
        # Idle / no-target: BeamRider needs FIRE to advance levels; default NOOP.
        game_a = jnp.where(target == no_target, idle_action, game_a)
        game_a = jnp.where(is_noop, 0, game_a)
        return game_a.astype(jnp.int32)

    def run_episode(rng):
        rng, reset_key = jax.random.split(rng)
        obs, state = env.reset(reset_key)
        prev_target = no_target
        mission_target = no_target
        persist_steps = jnp.array(0, dtype=jnp.int32)
        suspended_mission = jnp.array(no_target, dtype=jnp.int32)

        def cond(carry):
            (
                _state,
                _obs,
                _rng,
                _ret,
                done,
                _prev,
                _mission,
                _steps,
                _suspended,
                step,
                _metrics,
            ) = carry
            return jnp.logical_and(step < max_frames, jnp.logical_not(done))

        def body(carry):
            (
                state,
                obs,
                rng,
                ret,
                done,
                prev_target,
                mission_target,
                persist_steps,
                suspended_mission,
                step,
                metrics,
            ) = carry
            rng, step_key = jax.random.split(rng)

            if use_meta:
                target, persist_steps, suspended_mission = _jax_select_target_with_meta(
                    obs,
                    config,
                    meta_network,
                    meta_params,
                    meta_mask_hook,
                    forced_target_hook,
                    no_target,
                    last_target_idx=prev_target,
                    persist_steps_remaining=persist_steps,
                    suspended_mission_idx=suspended_mission,
                )
            elif heuristic_with_mission is not None:
                target, mission_target = heuristic_with_mission(
                    obs.raw_observation,
                    obs.objects,
                    prev_target,
                    mission_target,
                    step_key,
                )
                target = target.astype(jnp.int32)
                mission_target = mission_target.astype(jnp.int32)
                target = forced_target_hook(
                    jax.tree_util.tree_map(
                        lambda x: jnp.expand_dims(x, 0), obs.raw_observation
                    ),
                    jnp.expand_dims(target, 0),
                    config,
                )[0].astype(jnp.int32)
            else:
                target = jnp.ravel(
                    heuristic_target(
                        obs.raw_observation,
                        obs.objects,
                        prev_target,
                        step_key,
                    )
                )[0].astype(jnp.int32)
                target = forced_target_hook(
                    jax.tree_util.tree_map(
                        lambda x: jnp.expand_dims(x, 0), obs.raw_observation
                    ),
                    jnp.expand_dims(target, 0),
                    config,
                )[0].astype(jnp.int32)

            game_a = _skill_game_action(obs, target)
            new_obs, new_state, reward, step_done, _info = env.step(state, game_a)
            metrics = update_metrics(metrics, state, new_state, new_obs, reward)
            new_done = jnp.logical_or(done, step_done)
            target = jnp.where(new_done, no_target, target)
            mission_target = jnp.where(new_done, no_target, mission_target)
            persist_steps = jnp.where(new_done, jnp.array(0, dtype=jnp.int32), persist_steps)
            suspended_mission = jnp.where(
                new_done, jnp.array(no_target, dtype=jnp.int32), suspended_mission
            )
            return (
                new_state,
                new_obs,
                rng,
                ret + reward,
                new_done,
                target,
                mission_target,
                persist_steps,
                suspended_mission,
                step + 1,
                metrics,
            )

        init = (
            state,
            obs,
            rng,
            jnp.array(0.0),
            False,
            prev_target,
            mission_target,
            persist_steps,
            suspended_mission,
            0,
            init_episode_metrics(),
        )
        final = jax.lax.while_loop(cond, body, init)
        metrics = finalize_metrics(final[10], final[0])
        return final[3], metrics

    return jax.jit(run_episode)


def run_greedy_eval_episodes(
    config,
    params_tree=None,
    meta_params=None,
    *,
    num_episodes: int = 5,
    num_seeds: int = 3,
    base_seed: int | None = None,
    max_frames: int | None = None,
    progress: bool = True,
    on_episode_done=None,
) -> dict:
    """
    Greedy hierarchical eval on the fair eval wrapper (no sticky actions, no episodic life).

    Runs ``num_seeds * num_episodes`` full episodes and returns aggregate score stats.
    Meta is greedy argmax when ``meta_params`` is set; otherwise the env heuristic is used.

    Each episode is one JIT-compiled ``while_loop`` (not a slow Python step loop).
    """
    configure_ego_obs_from_config(config)
    configure_label_routing_from_config(config)

    num_skills = int(config["NUM_SKILLS"])
    ego_action_dim = int(config["EGO_ACTION_DIM"])
    hidden_size = int(config["HIDDEN_SIZE"])
    num_layers = int(config["NUM_LAYERS"])
    base_seed = int(config["SEED"] if base_seed is None else base_seed)
    restrict_fire_for_non_destroy = bool(
        config.get("RESTRICT_FIRE_FOR_NON_DESTROY", False)
    )
    # Match video rollout cap; 27000 Python steps × NN forward ≈ hours per env.
    max_frames = int(max_frames or config.get("EVAL_MAX_FRAMES", 2000))

    env, _renderer = create_eval_env(config)
    env_name = config.get("ENV_NAME", "Seaquest")
    manager = EnvManager()
    components = manager.get_components(env_name)

    skill_network = q_network_from_config(config)
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
    else:
        skill_vars = skill_network.init(
            jax.random.PRNGKey(base_seed + 1), dummy_ego, train=False
        )
        params_tree = jax.tree_util.tree_map(
            lambda p: jnp.stack([p] * num_skills, axis=0),
            skill_vars["params"],
        )

    run_episode = _make_jitted_greedy_episode_runner(
        env,
        skill_network,
        params_tree,
        config,
        components,
        meta_network=meta_network,
        meta_params=meta_params,
        max_frames=max_frames,
        num_skills=num_skills,
        ego_action_dim=ego_action_dim,
        restrict_fire_for_non_destroy=restrict_fire_for_non_destroy,
    )

    if progress:
        print(f"  Compiling greedy eval (max {max_frames} steps/episode)...", flush=True)
    _ = jax.block_until_ready(run_episode(jax.random.PRNGKey(base_seed + 999))[0])

    total_runs = num_seeds * num_episodes
    episode_returns: list[float] = []
    episode_metrics: list[dict[str, float]] = []
    run_idx = 0
    for seed_idx in range(num_seeds):
        for ep_idx in range(num_episodes):
            run_idx += 1
            if progress:
                print(
                    f"  Greedy eval episode {run_idx}/{total_runs} "
                    f"(seed {seed_idx + 1}/{num_seeds}, ep {ep_idx + 1}/{num_episodes})...",
                    flush=True,
                )
            ep_seed = base_seed + seed_idx * 10_000 + ep_idx * 997
            ep_return_j, ep_metrics_j = run_episode(jax.random.PRNGKey(ep_seed))
            ep_return = float(jax.device_get(ep_return_j))
            ep_metrics = metrics_to_python(jax.device_get(ep_metrics_j))
            episode_returns.append(ep_return)
            episode_metrics.append(ep_metrics)
            if on_episode_done is not None:
                on_episode_done()
            if progress:
                extra = ""
                if "seaquest" in str(env_name).lower():
                    extra = (
                        f" kills={ep_metrics.get('env_kills', 0):.0f}"
                        f" rescues={ep_metrics.get('env_rescues', 0):.0f}"
                        f" divers_max={ep_metrics.get('env_divers_max', 0):.0f}"
                    )
                print(f"    return={ep_return:.2f}{extra}", flush=True)

    arr = np.asarray(episode_returns, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std()) if arr.size > 1 else 0.0,
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "returns": episode_returns,
        "episode_metrics": episode_metrics,
        "num_episodes": int(arr.size),
        "num_seeds": int(num_seeds),
        "episodes_per_seed": int(num_episodes),
        "max_frames_per_episode": max_frames,
        **aggregate_episode_metrics(episode_metrics),
    }


def run_eval_rollout(config, params_tree=None, meta_params=None, max_frames=None):
    """
    Run a single evaluation rollout with the hierarchical PQN-style agent:
    - Fixed heuristic meta-policy to choose a target object.
    - Shared skill Q-network (one set of weights per skill).
    - Ego-centric action space mapped back to Atari actions.

    Returns the same tensors that `video_recorder.video_callback` expects.
    """
    configure_ego_obs_from_config(config)
    configure_label_routing_from_config(config)
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
    frame_stride = max(1, int(config.get("VIDEO_FRAME_STRIDE", 1)))
    max_env_steps = int(max_frames) * frame_stride

    env, renderer = create_eval_env(config)

    # Environment-specific hooks (meta-policy, action mapping, NO_TARGET sentinel)
    env_name = config.get("ENV_NAME", "Seaquest")
    manager = EnvManager()
    components = manager.get_components(env_name)
    heuristic_target = resolve_heuristic_target_fn(config, env_name, components)
    heuristic_with_mission = resolve_heuristic_with_mission_fn(config, env_name)
    heuristic_with_state = resolve_heuristic_with_state_fn(config, env_name, components)
    no_target_idx = components.no_target_idx
    action_mapper, _ = resolve_action_mappers(components, config)
    forced_target_hook = components.forced_target_hook
    meta_mask_hook = components.meta_mask_hook
    reset_policy_on_life_lost = bool(
        getattr(components, "reset_policy_on_life_lost", False)
    )
    extract_life_lost = components.extract_life_lost
    use_legal_ego_mask = bool(config.get("EGO_LEGAL_ACTION_MASK", False))
    legal_ego_mask_fn = make_realized_ego_mask_fn(
        action_mapper,
        noop_action=int(components.ego_mapper_noop_action),
        fire_action=int(components.ego_mapper_fire_action),
        ego_action_dim=int(ego_action_dim),
    )

    meta_network = None
    if meta_params is not None:
        meta_network = create_meta_network(config)

    # Build the shared skill Q-network (one set of params per skill)
    skill_network = q_network_from_config(config)

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
        init_pixel = _skill_pixel_obs_from_obs(config, obs)
        if init_pixel is not None:
            init_pixel = jnp.expand_dims(init_pixel, 0)
        network_vars = skill_network.init(
            jax.random.PRNGKey(seed + 1),
            init_x,
            train=False,
            target_idx=jnp.array(0, dtype=jnp.int32),
            pixel_obs=init_pixel,
            objects=obs.objects,
        )
        base_params = network_vars["params"]

        # Replicate the same initialization across all skills.
        params_tree = jax.tree_util.tree_map(
            lambda p: jnp.stack([p] * num_skills, axis=0),
            base_params,
        )

    # JIT skill Q eval once — the old per-step jax.vmap.apply was the reason
    # skills_only end videos appeared to hang after "Recording...".
    def _apply_all_skills(params_stack, ego_batch, target_idx_arr, pixel_batch, objects_batch):
        return apply_skill_q_all(
            skill_network,
            params_stack,
            ego_batch,
            target_idx=target_idx_arr,
            pixel_obs=pixel_batch,
            objects=objects_batch,
            stack_size=int(config.get("FRAME_STACK_SIZE", 4)),
            destroy_skill_idx=_destroy_skill_idx(config),
        )

    apply_all_skills = jax.jit(_apply_all_skills)

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
    mission_target_idx = no_target_idx
    policy_state = initial_policy_state(no_target_idx)
    persist_steps_remaining = jnp.array(0, dtype=jnp.int32)
    suspended_mission_idx = jnp.array(no_target_idx, dtype=jnp.int32)
    print(
        f"Eval rollout: up to {max_env_steps} env steps, "
        f"log every {frame_stride} → ≤{max_frames} video frames..."
    )
    for step_i in range(max_env_steps):
        if step_i > 0 and step_i % max(100 * frame_stride, 100) == 0:
            print(
                f"  eval step {step_i}/{max_env_steps} "
                f"(logged {len(states_list)}/{max_frames})..."
            )
        rng, step_key = jax.random.split(rng)

        if meta_network is None:
            if heuristic_with_state is not None:
                target_obj_idx, policy_state = heuristic_with_state(
                    obs.raw_observation,
                    obs.objects,
                    jnp.array(prev_target_idx, dtype=jnp.int32),
                    policy_state,
                    step_key,
                )
                target_obj_idx = int(jnp.ravel(target_obj_idx)[0])
            elif heuristic_with_mission is not None:
                target_obj_idx, mission_target_idx = heuristic_with_mission(
                    obs.raw_observation,
                    obs.objects,
                    jnp.array(prev_target_idx, dtype=jnp.int32),
                    jnp.array(mission_target_idx, dtype=jnp.int32),
                    step_key,
                )
                target_obj_idx = int(jnp.ravel(target_obj_idx)[0])
                mission_target_idx = int(jnp.ravel(mission_target_idx)[0])
            else:
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
            target_obj_idx, persist_steps_remaining, suspended_mission_idx = _select_target_with_meta(
                obs,
                config,
                meta_network,
                meta_params,
                meta_mask_hook,
                forced_target_hook,
                no_target_idx,
                last_target_idx=prev_target_idx,
                persist_steps_remaining=persist_steps_remaining,
                suspended_mission_idx=suspended_mission_idx,
            )
            target_obj_idx = int(target_obj_idx)
        prev_target_idx = target_obj_idx

        is_meta_noop = meta_use_noop_target and target_obj_idx == meta_noop_target_idx
        is_meta_fire = meta_use_fire_target and target_obj_idx == meta_fire_target_idx
        if is_meta_noop:
            global_action = 0
            active_skill_idx = 0
            combined_q = np.zeros(num_skills, dtype=np.float32)
            distance = np.nan
            angle = np.nan
        elif target_obj_idx == no_target_idx:
            global_action = int(config.get("NO_TARGET_GAME_ACTION", 0))
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
            ego_obs = _skill_input_from_obs(
                config,
                obs.raw_observation,
                obs.objects,
                jnp.array(target_obj_idx, dtype=jnp.int32),
            )
            ego_obs_batch = ego_obs[jnp.newaxis, ...]
            pixel_obs = _skill_pixel_obs_from_obs(config, obs)

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
                distance = _hud_target_distance_px(
                    obs.raw_observation,
                    obs.objects,
                    int(target_obj_idx),
                    last_frame_ego,
                )
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
            target_idx_arr = jnp.asarray([target_obj_idx], dtype=jnp.int32)
            pixel_batch = (
                jnp.expand_dims(pixel_obs, 0) if pixel_obs is not None else None
            )
            q_vals = apply_all_skills(
                params_tree,
                ego_obs_batch,
                target_idx_arr,
                pixel_batch,
                obs.objects,
            )  # [num_skills, 1, ego_action_dim]
            q_vals = jnp.squeeze(q_vals, axis=1)  # [num_skills, ego_action_dim]

            combined_q = np.array(jnp.max(q_vals, axis=-1))

            fire_indices = fire_ego_action_indices(ego_action_dim)
            q_active = q_vals[active_skill_idx]
            if restrict_fire_for_non_destroy and active_skill_idx != 0:
                q_active = q_active.at[fire_indices].set(-jnp.inf)
            if use_legal_ego_mask:
                legal = legal_ego_mask_fn(
                    obs.raw_observation,
                    jnp.array(target_obj_idx, dtype=jnp.int32),
                    obs.objects,
                )
                q_active = mask_q_with_legal_ego(q_active, legal)

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
        prev_env_state = state
        obs, state, _reward, done, _info = env.step(state, global_action)
        obs, state = jax.device_get((obs, state))
        done = bool(done)
        life_lost = 0
        if reset_policy_on_life_lost:
            life_lost = int(
                jax.device_get(extract_life_lost(prev_env_state, state))
            )

        # Keep video length = VIDEO_MAX_FRAMES; subsample env steps via stride.
        log_frame = (step_i % frame_stride == 0) or done
        if log_frame:
            if len(states_list) >= max_frames:
                break
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
            # Keep filling the video budget across lives / episodes (demo rollouts).
            rng, rng_reset = jax.random.split(rng)
            obs, state = env.reset(rng_reset)
            obs, state = jax.device_get((obs, state))
            prev_target_idx = no_target_idx
            mission_target_idx = no_target_idx
            policy_state = initial_policy_state(no_target_idx)
            persist_steps_remaining = jnp.array(0, dtype=jnp.int32)
            suspended_mission_idx = jnp.array(no_target_idx, dtype=jnp.int32)
        elif life_lost > 0:
            # Kangaroo etc.: life loss respawns in-env without done=True.
            prev_target_idx = no_target_idx
            mission_target_idx = no_target_idx
            policy_state = initial_policy_state(no_target_idx)
            persist_steps_remaining = jnp.array(0, dtype=jnp.int32)
            suspended_mission_idx = jnp.array(no_target_idx, dtype=jnp.int32)

    print(
        f"Eval rollout done ({len(states_list)} video frames "
        f"from ≤{max_env_steps} env steps, stride={frame_stride})."
    )
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
    display_backend: str = "pygame",
    pygame_screen=None,
    pygame_font=None,
):
    """
    Run a live, interactive visualization loop.

    Default display backend is Pygame. Prefer calling ``bootstrap_pygame_window``
    before loading JAX weights so the SDL window can map; pass the resulting
    screen/font via ``pygame_screen`` / ``pygame_font``. Use
    ``display_backend="opencv"`` if Pygame still fails to show a window.
    """
    def _log(msg: str) -> None:
        print(f"[live-viz] {msg}", flush=True)

    configure_ego_obs_from_config(config)
    configure_label_routing_from_config(config)
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
    # VIDEO_MAX_FRAMES is for recorded rollouts; live mode runs until quit unless capped.
    frame_limit = max_frames
    display_fps = int(fps if fps is not None else config.get("VISUALIZER_FPS", 15))
    display_backend = str(display_backend or "pygame").lower().strip()
    if display_backend not in ("opencv", "pygame"):
        raise ValueError(f"Unknown display_backend={display_backend!r} (use opencv|pygame)")

    _log(f"JAX backend={jax.default_backend()} devices={jax.devices()}")
    _log(f"display_backend={display_backend}")

    window_name = "Hierarchical PQN Live Visualization"
    scale = 3
    window_size = (160 * scale, 210 * scale)
    screen = pygame_screen
    font = pygame_font
    clock = None
    wait_ms = max(1, int(round(1000.0 / max(display_fps, 1))))
    cv2 = None

    if display_backend == "pygame":
        if screen is None or font is None:
            screen, font = bootstrap_pygame_window(window_name, window_size)
        else:
            pygame.display.set_caption(window_name)
            if screen.get_size() != window_size:
                screen = pygame.display.set_mode(
                    window_size, pygame.SHOWN | pygame.RESIZABLE
                )
            pygame.display.flip()
            pygame.event.pump()
            _raise_pygame_window()
    else:
        # Import OpenCV only for this backend — importing cv2 before pygame
        # display init can leave SDL on the dummy driver (no window).
        import cv2 as _cv2

        cv2 = _cv2
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_size[0], window_size[1])

    def _show_status(text: str) -> None:
        if display_backend == "pygame":
            screen.fill((20, 20, 20))
            surf = font.render(text, True, (240, 240, 240))
            screen.blit(surf, (16, window_size[1] // 2 - 10))
            pygame.display.flip()
            pygame.event.pump()
            return
        canvas = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)
        cv2.putText(
            canvas,
            text[:90],
            (16, window_size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, canvas)
        cv2.waitKey(1)

    def _show_rgb_frame(frame_rgb: np.ndarray, overlay: str | None = None) -> None:
        """Show an HxWx3 RGB uint8 frame scaled to window_size."""
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)
        if not frame_rgb.flags["C_CONTIGUOUS"]:
            frame_rgb = np.ascontiguousarray(frame_rgb)
        if display_backend == "pygame":
            surf = pygame.surfarray.make_surface(
                np.ascontiguousarray(frame_rgb.transpose(1, 0, 2))
            )
            surf = pygame.transform.scale(surf, window_size)
            screen.blit(surf, (0, 0))
            if overlay:
                text_surface = font.render(overlay, True, (255, 255, 255))
                tx = (window_size[0] - text_surface.get_width()) // 2
                ty = window_size[1] - text_surface.get_height() - 10
                screen.blit(text_surface, (tx, ty))
            pygame.display.flip()
            pygame.event.pump()
            return
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, window_size, interpolation=cv2.INTER_NEAREST)
        if overlay:
            cv2.putText(
                bgr,
                overlay,
                (8, window_size[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(window_name, bgr)
        cv2.waitKey(1)

    def _poll_input() -> str | None:
        """Poll UI input. Returns 'quit', 'reset', or None."""
        if display_backend == "pygame":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_q,
                    pygame.K_ESCAPE,
                ):
                    return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    return "reset"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    debug_heuristic_snapshot(
                        obs.raw_observation,
                        obs.objects,
                        jnp.array(prev_target_idx, dtype=jnp.int32),
                    )
            return None
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return "quit"
        if key in (ord("r"), ord("R")):
            return "reset"
        return None

    _show_status("Creating environment...")
    _log("Creating eval env...")
    env, renderer = create_eval_env(config)

    # Environment-specific hooks (meta-policy, action mapping, NO_TARGET sentinel)
    env_name = config.get("ENV_NAME", "Seaquest")
    manager = EnvManager()
    components = manager.get_components(env_name)
    heuristic_target = resolve_heuristic_target_fn(config, env_name, components)
    heuristic_with_mission = resolve_heuristic_with_mission_fn(config, env_name)
    heuristic_with_state = resolve_heuristic_with_state_fn(config, env_name, components)
    no_target_idx = components.no_target_idx
    action_mapper, _ = resolve_action_mappers(components, config)
    forced_target_hook = components.forced_target_hook
    meta_mask_hook = components.meta_mask_hook
    reset_policy_on_life_lost = bool(
        getattr(components, "reset_policy_on_life_lost", False)
    )
    extract_life_lost = components.extract_life_lost
    use_legal_ego_mask = bool(config.get("EGO_LEGAL_ACTION_MASK", False))
    legal_ego_mask_fn = make_realized_ego_mask_fn(
        action_mapper,
        noop_action=int(components.ego_mapper_noop_action),
        fire_action=int(components.ego_mapper_fire_action),
        ego_action_dim=int(ego_action_dim),
    )

    meta_network = None
    if meta_params is not None:
        meta_network = create_meta_network(config)

    skill_network = q_network_from_config(config)

    rng = jax.random.PRNGKey(seed)
    rng, rng_reset = jax.random.split(rng)
    _show_status("Compiling env.reset (first call can take a while)...")
    _log("Resetting env (compiling)...")
    obs, state = env.reset(rng_reset)
    obs, state = jax.device_get((obs, state))
    _log("env.reset ready")

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
        init_pixel = _skill_pixel_obs_from_obs(config, obs)
        if init_pixel is not None:
            init_pixel = jnp.expand_dims(init_pixel, 0)
        network_vars = skill_network.init(
            jax.random.PRNGKey(seed + 1),
            init_x,
            train=False,
            target_idx=jnp.array(0, dtype=jnp.int32),
            pixel_obs=init_pixel,
            objects=obs.objects,
        )
        base_params = network_vars["params"]
        params_tree = jax.tree_util.tree_map(
            lambda p: jnp.stack([p] * num_skills, axis=0),
            base_params,
        )

    # JIT skill Q eval once (same fix as record-mode eval rollouts).
    def _apply_all_skills(params_stack, ego_batch, target_idx_arr, pixel_batch, objects_batch):
        return apply_skill_q_all(
            skill_network,
            params_stack,
            ego_batch,
            target_idx=target_idx_arr,
            pixel_obs=pixel_batch,
            objects=objects_batch,
            stack_size=int(config.get("FRAME_STACK_SIZE", 4)),
            destroy_skill_idx=_destroy_skill_idx(config),
        )

    apply_all_skills = jax.jit(_apply_all_skills)

    _show_status("Compiling renderer...")
    _log("Rendering first frame (compiling)...")
    base_state = unwrap_base_env_state(state.atari_state)
    sample_frame = renderer.render(base_state)
    sample_frame = np.array(jax.device_get(sample_frame), dtype=np.uint8)
    if sample_frame.ndim == 2:
        sample_frame = np.stack([sample_frame] * 3, axis=-1)
    H, W, _ = sample_frame.shape
    window_size = (W * scale, H * scale)
    if display_backend == "pygame":
        if screen.get_size() != window_size:
            screen = pygame.display.set_mode(
                window_size, pygame.SHOWN | pygame.RESIZABLE
            )
        font = pygame.font.Font(None, 20)
        clock = pygame.time.Clock()
        _raise_pygame_window()
    else:
        cv2.resizeWindow(window_name, window_size[0], window_size[1])

    _show_rgb_frame(sample_frame, overlay="Warming up first step compiles...")
    _log("First frame shown in window")

    running = True
    prev_target_idx = no_target_idx
    mission_target_idx = no_target_idx
    policy_state = initial_policy_state(no_target_idx)
    persist_steps_remaining = jnp.array(0, dtype=jnp.int32)
    suspended_mission_idx = jnp.array(no_target_idx, dtype=jnp.int32)
    frame_count = 0

    def reset_env_episode():
        nonlocal obs, state, prev_target_idx, mission_target_idx, policy_state
        nonlocal persist_steps_remaining, suspended_mission_idx, rng
        rng_local, rng_reset_local = jax.random.split(rng)
        obs_local, state_local = env.reset(rng_reset_local)
        obs = obs_local
        state = state_local
        rng = rng_local
        prev_target_idx = no_target_idx
        mission_target_idx = no_target_idx
        policy_state = initial_policy_state(no_target_idx)
        persist_steps_remaining = jnp.array(0, dtype=jnp.int32)
        suspended_mission_idx = jnp.array(no_target_idx, dtype=jnp.int32)

    def _select_target(step_key):
        nonlocal mission_target_idx, persist_steps_remaining, suspended_mission_idx
        nonlocal policy_state
        if meta_network is None:
            if heuristic_with_state is not None:
                target_obj_idx, policy_state = heuristic_with_state(
                    obs.raw_observation,
                    obs.objects,
                    jnp.array(prev_target_idx, dtype=jnp.int32),
                    policy_state,
                    step_key,
                )
                target_obj_idx = int(jnp.ravel(target_obj_idx)[0])
            elif heuristic_with_mission is not None:
                target_obj_idx, mission_target_idx = heuristic_with_mission(
                    obs.raw_observation,
                    obs.objects,
                    jnp.array(prev_target_idx, dtype=jnp.int32),
                    jnp.array(mission_target_idx, dtype=jnp.int32),
                    step_key,
                )
                target_obj_idx = int(jnp.ravel(target_obj_idx)[0])
                mission_target_idx = int(jnp.ravel(mission_target_idx)[0])
            else:
                target_obj_idx = heuristic_target(
                    obs.raw_observation,
                    obs.objects,
                    jnp.asarray(prev_target_idx, dtype=jnp.int32),
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
            return target_obj_idx
        target_obj_idx, persist_steps_remaining, suspended_mission_idx = _select_target_with_meta(
            obs,
            config,
            meta_network,
            meta_params,
            meta_mask_hook,
            forced_target_hook,
            no_target_idx,
            last_target_idx=prev_target_idx,
            persist_steps_remaining=persist_steps_remaining,
            suspended_mission_idx=suspended_mission_idx,
        )
        return int(target_obj_idx)

    skill_names = [str(n) for n in (config.get("SKILL_NAMES") or [])]
    _ego_action_names = {a.value: a.name for a in EgocentricAction}

    def _policy_action(target_obj_idx):
        is_meta_noop = meta_use_noop_target and target_obj_idx == meta_noop_target_idx
        is_meta_fire = meta_use_fire_target and target_obj_idx == meta_fire_target_idx
        if target_obj_idx == no_target_idx or is_meta_noop:
            return 0, 0, 0, np.nan, np.nan
        if is_meta_fire:
            return meta_fire_game_action, 0, 1, np.nan, np.nan

        raw_label = obs.objects.semantic_label[-1, target_obj_idx]
        active_skill_idx = int(label_to_skill_idx(raw_label, num_skills))

        ego_obs = _skill_input_from_obs(
            config,
            obs.raw_observation,
            obs.objects,
            jnp.array(target_obj_idx, dtype=jnp.int32),
        )
        ego_obs_batch = ego_obs[jnp.newaxis, ...]
        pixel_obs = _skill_pixel_obs_from_obs(config, obs)

        channels_per_frame = ego_obs_channels_per_frame()
        last_frame_ego = ego_obs[-channels_per_frame:]
        if using_legacy_ego_obs():
            dx_pix = float(last_frame_ego[0] * SCREEN_WIDTH)
            dy_pix = float(last_frame_ego[1] * SCREEN_HEIGHT)
            distance = float(jnp.sqrt(dx_pix ** 2 + dy_pix ** 2))
            angle = float(jnp.arctan2(dy_pix, dx_pix))
        else:
            distance = _hud_target_distance_px(
                obs.raw_observation,
                obs.objects,
                int(target_obj_idx),
                last_frame_ego,
            )
            player = _get_player(obs.raw_observation)
            y_is_bottom_edge = _uses_bottom_edge_y(obs.raw_observation)
            p_cx = float(player.x[-1] + player.width[-1] / 2.0)
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

        pixel_batch = jnp.expand_dims(pixel_obs, 0) if pixel_obs is not None else None
        q_vals = apply_all_skills(
            params_tree,
            ego_obs_batch,
            jnp.asarray([target_obj_idx], dtype=jnp.int32),
            pixel_batch,
            obs.objects,
        )
        q_vals = jnp.squeeze(q_vals, axis=1)

        fire_indices = fire_ego_action_indices(ego_action_dim)
        q_active = q_vals[active_skill_idx]
        if restrict_fire_for_non_destroy and active_skill_idx != 0:
            q_active = q_active.at[fire_indices].set(-jnp.inf)
        if use_legal_ego_mask:
            legal = legal_ego_mask_fn(
                obs.raw_observation,
                jnp.array(target_obj_idx, dtype=jnp.int32),
                obs.objects,
            )
            q_active = mask_q_with_legal_ego(q_active, legal)

        chosen_ego = int(jnp.argmax(q_active))
        global_action = int(
            action_mapper(
                chosen_ego,
                obs.raw_observation,
                jnp.array(target_obj_idx, dtype=jnp.int32),
                obs.objects,
            )
        )
        return global_action, active_skill_idx, chosen_ego, distance, angle

    # Warm first-frame compiles while the window stays responsive/visible.
    _show_status("Compiling heuristic / policy / env.step...")
    _log("Warmup: heuristic + policy + env.step (can take a bit)...")
    rng, warm_key = jax.random.split(rng)
    warm_target = _select_target(warm_key)
    warm_action, _, _, _, _ = _policy_action(warm_target)
    obs, state, _reward, _done, _info = env.step(state, int(warm_action))
    obs, state = jax.device_get((obs, state))
    prev_target_idx = warm_target
    _log("Warmup done — entering live loop")
    _log(f"Ready — live loop at {display_fps} FPS (q/esc quit, r reset)")

    while running and (frame_limit is None or frame_count < frame_limit):
        action = _poll_input()
        if action == "quit":
            running = False
            break
        if action == "reset":
            reset_env_episode()

        if frame_count < 5:
            _log(f"frame {frame_count}: select target...")
        rng, step_key = jax.random.split(rng)
        target_obj_idx = _select_target(step_key)
        prev_target_idx = target_obj_idx

        if frame_count < 5:
            _log(f"frame {frame_count}: policy action (target={target_obj_idx})...")
        global_action, active_skill_idx, chosen_ego, distance, angle = _policy_action(
            target_obj_idx
        )
        skill_name = (
            skill_names[active_skill_idx]
            if 0 <= active_skill_idx < len(skill_names)
            else str(active_skill_idx)
        )
        ego_name = _ego_action_names.get(chosen_ego, str(chosen_ego))

        if frame_count < 5:
            _log(
                f"frame {frame_count}: env.step(action={global_action}) "
                f"skill={skill_name} ego={ego_name}..."
            )
        # Step env
        prev_env_state = state
        obs, state, _reward, done, _info = env.step(state, global_action)
        obs, state = jax.device_get((obs, state))
        done = bool(done)
        life_lost = 0
        if reset_policy_on_life_lost:
            life_lost = int(
                jax.device_get(extract_life_lost(prev_env_state, state))
            )

        if frame_count < 5:
            _log(f"frame {frame_count}: render...")
        # Render current frame
        base_state = unwrap_base_env_state(state.atari_state)
        frame = renderer.render(base_state)
        frame = np.array(jax.device_get(frame), dtype=np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        # Add a red border around the selected target object (same style as videos).
        draw_viz_outline = bool(config.get("VISUALIZER_DRAW_TARGET_OUTLINE", True))
        if draw_viz_outline and target_obj_idx != no_target_idx:
            ox = np.array(obs.objects.x[-1])
            oy = np.array(obs.objects.y[-1])
            ow = np.array(obs.objects.width[-1])
            oh = np.array(obs.objects.height[-1])
            oactive = np.array(obs.objects.active[-1])
            olabels = np.array(obs.objects.semantic_label[-1])
            outline_min_px = 1
            if (
                0 <= int(target_obj_idx) < len(olabels)
                and int(olabels[int(target_obj_idx)]) == LABEL_AVOID
            ):
                outline_min_px = 10

            frame_chw = np.transpose(frame, (2, 0, 1))
            frame_chw = draw_target_outline(
                frame_chw,
                target_obj_idx,
                ox,
                oy,
                ow,
                oh,
                oactive,
                min_game_px=outline_min_px,
            )
            frame = np.transpose(frame_chw, (1, 2, 0))

        overlay = f"{skill_name} ego={ego_name} game={global_action}"
        if not np.isnan(distance):
            overlay = f"{overlay} d={distance:.1f}px a={angle:.2f}"
        if frame_count < 5:
            _log(f"frame {frame_count}: display...")
        _show_rgb_frame(frame, overlay=overlay)
        if display_backend == "pygame":
            clock.tick(display_fps)
        frame_count += 1
        if frame_count < 5:
            _log(f"frame {frame_count - 1}: done")

        if done:
            _log("episode done — env + waypoint plan reset")
            rng, rng_reset = jax.random.split(rng)
            obs, state = env.reset(rng_reset)
            obs, state = jax.device_get((obs, state))
            prev_target_idx = no_target_idx
            mission_target_idx = no_target_idx
            policy_state = initial_policy_state(no_target_idx)
            persist_steps_remaining = jnp.array(0, dtype=jnp.int32)
            suspended_mission_idx = jnp.array(no_target_idx, dtype=jnp.int32)
        elif life_lost > 0:
            # Kangaroo: lives drop on crash without done=True; restart the climb plan.
            _log(f"life lost (livesΔ={life_lost}) — waypoint plan reset")
            prev_target_idx = no_target_idx
            mission_target_idx = no_target_idx
            policy_state = initial_policy_state(no_target_idx)
            persist_steps_remaining = jnp.array(0, dtype=jnp.int32)
            suspended_mission_idx = jnp.array(no_target_idx, dtype=jnp.int32)

    if frame_limit is not None and frame_count >= frame_limit:
        print(f"Live visualization stopped after {frame_count} frames (--max-frames={frame_limit}).")
    if display_backend == "pygame":
        pygame.quit()
    else:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)


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
        "--remap-navigate-to-collect",
        action="store_true",
        default=False,
        help=(
            "Route LABEL_NAVIGATE targets through the COLLECT skill head "
            "(same fold as NUM_SKILLS<4). Useful for transfer smoke tests "
            "when the NAVIGATE head fails to generalize."
        ),
    )
    parser.add_argument(
        "--no-outline",
        action="store_true",
        default=False,
        help=(
            "Disable target-outline painting for pixel_embed CNN input and skip "
            "the red target box overlay in live/recorded visualization."
        ),
    )
    parser.add_argument(
        "--black-pixel",
        action="store_true",
        default=False,
        help=(
            "Zero out pixel_obs before the skill CNN (ablation: policy without vision)."
        ),
    )
    parser.add_argument(
        "--heuristic-mode",
        type=str,
        default=None,
        help="Override HEURISTIC_MODE (e.g. nearest for RQ2 transfer eval).",
    )
    parser.add_argument(
        "--video-out",
        type=str,
        default=None,
        help=(
            "When --record is set, also write an mp4 to this path "
            "(works with --wandb-mode disabled)."
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
        "--display",
        type=str,
        choices=["opencv", "pygame"],
        default="pygame",
        help=(
            "Live window backend. Default pygame. "
            "Use opencv if pygame fails to map a window once JAX threads are running."
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

    # Open the pygame window before JAX weight loading so SDL can map it.
    pygame_screen = None
    pygame_font = None
    if (not args.record) and str(args.display).lower().strip() == "pygame":
        pygame_screen, pygame_font = bootstrap_pygame_window()

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

    # Checkpoint *_config.yaml may set WANDB_MODE=online; keep the CLI default
    # (disabled) authoritative for local visualization.
    if args.wandb_mode is not None:
        base_config["WANDB_MODE"] = args.wandb_mode

    # Keep explicit CLI env override authoritative for cross-env transfer runs,
    # even when checkpoint-side *_config.yaml is present.
    if args.env_name is not None:
        base_config["ENV_NAME"] = args.env_name
    if args.remap_navigate_to_collect:
        base_config["REMAP_NAVIGATE_TO_COLLECT"] = True
        print(
            "REMAP_NAVIGATE_TO_COLLECT=True — NAVIGATE waypoints use the COLLECT skill head"
        )
    apply_visualizer_config_overrides(
        base_config,
        no_outline=bool(args.no_outline),
        black_pixel=bool(args.black_pixel),
        heuristic_mode=args.heuristic_mode,
    )
    if args.no_outline:
        print("SKILL_PIXEL_PAINT_TARGET_OUTLINE=False — no target ring for CNN or viz overlay")
    if args.black_pixel:
        print("SKILL_PIXEL_BLACKOUT=True — pixel_obs zeroed before skill CNN")

    # If requested, run live interactive visualization and exit.
    if not args.record:
        run_live_visualization(
            base_config,
            params_tree=params_tree,
            meta_params=meta_params,
            max_frames=args.max_frames,
            fps=args.fps,
            display_backend=args.display,
            pygame_screen=pygame_screen,
            pygame_font=pygame_font,
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
    video_out = args.video_out
    if video_out is None and base_config.get("WANDB_MODE", "disabled") == "disabled":
        video_out = str(
            Path("outputs")
            / "videos"
            / f"{base_config.get('ENV_NAME', 'env')}_transfer_viz.mp4"
        )
        print(f"No --video-out set; writing local mp4 to {video_out}")
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
        video_output_path=video_out,
        draw_target_outline_overlay=bool(
            base_config.get("VISUALIZER_DRAW_TARGET_OUTLINE", True)
        ),
    )
    wait_for_video_finish()

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()