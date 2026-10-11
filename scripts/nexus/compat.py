"""Runtime setup so NEXUS trains/evals against the installed Seaquest fork.

Default (cluster): use ``JAXATARI_ROOT`` / the installed ``jaxtari`` package
(the robustness Seaquest fork). NEXUS's vendored copy is only used with
``--vendored-seaquest``.

Also:
- restores ``reward_funcs`` / ``all_rewards`` hooks NEXUS skill rewards need
- maps NEXUS ``DisableEnemiesWrapper`` → robustness ``no_enemies`` mod body
- strips a broken ``etils`` import in NEXUS seaquest rewards
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import jax.numpy as jnp


def apply_nexus_source_fixes(nexus_root: Path) -> None:
    """Patch NEXUS sources in-place (idempotent)."""
    path = (
        nexus_root
        / "src"
        / "symbolic_options"
        / "reward_functions"
        / "seaquest.py"
    )
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        bad = "from etils.exm.dummy import curr_job_name\nfrom flax.linen import meta\n"
        if text.startswith(bad):
            path.write_text(text[len(bad) :], encoding="utf-8")
            print(f"Patched dead imports in {path}")

    for pkg in ("jaxtari", "jaxatari"):
        wrappers = nexus_root / "JAXAtari" / "src" / pkg / "wrappers.py"
        if wrappers.is_file():
            text = wrappers.read_text(encoding="utf-8")
            new = text.replace('print("len obs: ", len(obs))', "pass  # CBL: struct obs")
            new = new.replace('print("obs: ", obs)', "pass  # CBL")
            new = new.replace("print(obs)", "pass  # CBL")
            if new != text:
                wrappers.write_text(new, encoding="utf-8")
                print(f"Patched len(obs) debug prints in {wrappers}")

    # Modern jaxtari nests SeaquestState under ObjectCentricState.atari_state;
    # NEXUS video_recorder only unwraps MultiRewardLog / Pixel / Atari.
    video = (
        nexus_root
        / "src"
        / "symbolic_options"
        / "utils"
        / "video_recorder.py"
    )
    if video.is_file():
        text = video.read_text(encoding="utf-8")
        needle = (
            "    if isinstance(states, PixelState):\n"
            "        states = states.atari_state\n\n"
            "    if isinstance(states, AtariState):\n"
            "        states = states.env_state\n"
        )
        insert = (
            "    if isinstance(states, PixelState):\n"
            "        states = states.atari_state\n\n"
            "    # CBL: modern jaxtari OC / Pixel+OC nest AtariState one level deeper\n"
            "    while hasattr(states, \"atari_state\") and not isinstance(states, AtariState):\n"
            "        states = states.atari_state\n\n"
            "    if isinstance(states, AtariState):\n"
            "        states = states.env_state\n"
        )
        if needle in text and "CBL: modern jaxtari OC" not in text:
            text = text.replace(needle, insert, 1)
            old_fallback = "        num_states = len(states[-1])\n"
            new_fallback = (
                "        num_states = int(jax.tree_util.tree_leaves(states)[0].shape[0])\n"
            )
            if old_fallback in text:
                text = text.replace(old_fallback, new_fallback, 1)
            video.write_text(text, encoding="utf-8")
            print(f"Patched modern OC state unwrap in {video}")

    _patch_main_hrl_factory_fallback(nexus_root)


_CBL_HRL_MAIN_MARKER = "# CBL: NO_REWARDS HRL factory fallback"
_CBL_HRL_MAIN_BLOCK = '''
    else:
        # CBL: NO_REWARDS HRL factory fallback for TimePilot / NameThisGame
        # (scripts/nexus/hrl_env.py — symbolic NEXUS stays Seaquest-only).
        import sys as _cbl_sys
        from pathlib import Path as _CblPath
        _cbl_scripts = _CblPath(__file__).resolve().parents[2] / "scripts"
        # When run via CBL run_nexus runner, PROJECT_ROOT/scripts is already on path.
        if str(_cbl_scripts) not in _cbl_sys.path:
            # Fallback: walk up from NEXUS_ROOT to ConceptBasedLearning/scripts
            _nexus_root = _CblPath(__file__).resolve().parents[1]
            for _cand in (
                _nexus_root.parent.parent / "scripts",  # third_party/symbolic_options → CBL
                _nexus_root.parent / "scripts",
            ):
                if (_cand / "nexus" / "hrl_env.py").is_file():
                    _cbl_sys.path.insert(0, str(_cand))
                    break
        from nexus.hrl_env import try_build_hrl_no_rewards_setup
        _cbl_setup = try_build_hrl_no_rewards_setup(config)
        if _cbl_setup is None:
            raise NotImplementedError(f"Env {config['ENV_NAME']} not implemented.")
        env = _cbl_setup["env"]
        eval_envs = _cbl_setup["eval_envs"]
        create_env = _cbl_setup["create_env"]
        renderer = _cbl_setup["renderer"]
        reward_funcs = _cbl_setup["reward_funcs"]
        learned_meta_policy = _cbl_setup["learned_meta_policy"]
        llm_meta_policy = _cbl_setup["llm_meta_policy"]
        conditional_meta_policy = _cbl_setup["conditional_meta_policy"]
        combined_meta_policy = _cbl_setup["combined_meta_policy"]
        test_env = _cbl_setup["test_env"]
        test_env_modif = _cbl_setup["test_env_modif"]
        env_params = _cbl_setup["env_params"]
        basic_env = _cbl_setup["basic_env"]
'''


def _patch_main_hrl_factory_fallback(nexus_root: Path) -> None:
    """Replace ``build_setup``'s final NotImplementedError with CBL HRL factory."""
    main_py = nexus_root / "main.py"
    if not main_py.is_file():
        return
    text = main_py.read_text(encoding="utf-8")
    if _CBL_HRL_MAIN_MARKER in text:
        return
    old = (
        "    else:\n"
        "        raise NotImplementedError(f\"Env {config['ENV_NAME']} not implemented.\")\n"
    )
    # Also match double-quoted / slightly different spacing from upstream.
    old_alt = (
        '    else:\n'
        '        raise NotImplementedError(f"Env {config[\'ENV_NAME\']} not implemented.")\n'
    )
    if old in text:
        text = text.replace(old, _CBL_HRL_MAIN_BLOCK.lstrip("\n"), 1)
    elif old_alt in text:
        text = text.replace(old_alt, _CBL_HRL_MAIN_BLOCK.lstrip("\n"), 1)
    else:
        # Last-resort: find the raise line inside build_setup.
        needle = 'raise NotImplementedError(f"Env {config[\'ENV_NAME\']} not implemented.")'
        needle2 = 'raise NotImplementedError(f"Env {config[\'ENV_NAME\']} not implemented.")'
        # Try common forms
        for n in (
            'raise NotImplementedError(f"Env {config[\'ENV_NAME\']} not implemented.")',
            "raise NotImplementedError(f\"Env {config['ENV_NAME']} not implemented.\")",
        ):
            if n in text and "Craftax" in text:
                # Replace the else+raise block heuristically
                idx = text.rfind("    else:\n", 0, text.find(n) + 1)
                if idx < 0:
                    continue
                end = text.find("\n", text.find(n)) + 1
                text = text[:idx] + _CBL_HRL_MAIN_BLOCK.lstrip("\n") + text[end:]
                break
        else:
            print(f"WARNING: could not patch HRL factory into {main_py}")
            return
    main_py.write_text(text, encoding="utf-8")
    print(f"Patched CBL HRL factory fallback into {main_py}")


def prepend_sys_path(paths: Sequence[Path]) -> None:
    for p in reversed([Path(x).resolve() for x in paths]):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


def _clear_jaxtari_modules() -> None:
    to_drop = [
        k
        for k in sys.modules
        if k in ("jaxtari", "jaxatari")
        or k.startswith("jaxtari.")
        or k.startswith("jaxatari.")
    ]
    for k in to_drop:
        del sys.modules[k]


def install_jaxatari(src: Path, *, label: str) -> Path:
    src = src.resolve()
    prepend_sys_path([src])
    _clear_jaxtari_modules()
    import jaxtari  # noqa: F401

    print(f"Using jaxtari from {src} ({label})")
    return src


def ensure_reward_funcs_api() -> None:
    """Monkeypatch JaxSeaquest so NEXUS can pass reward_funcs=..."""
    import jax
    from jax.flatten_util import ravel_pytree
    from jaxtari.games.jax_seaquest import JaxSeaquest

    if getattr(JaxSeaquest, "_cbl_nexus_reward_funcs_patched", False):
        return

    _orig_init = JaxSeaquest.__init__
    _orig_step = JaxSeaquest.step

    def __init__(self, consts=None, reward_funcs: list[Callable] | None = None, **kwargs):
        try:
            _orig_init(self, consts=consts, **kwargs)
        except TypeError:
            _orig_init(self, consts)
        self.reward_funcs = tuple(reward_funcs) if reward_funcs is not None else None
        if not hasattr(self, "frame_stack_size"):
            self.frame_stack_size = 4
        if not hasattr(self, "action_set") or self.action_set is None:
            from jaxtari.environment import JAXAtariAction as Action

            self.action_set = [
                Action.NOOP,
                Action.FIRE,
                Action.UP,
                Action.RIGHT,
                Action.LEFT,
                Action.DOWN,
                Action.UPRIGHT,
                Action.UPLEFT,
                Action.DOWNRIGHT,
                Action.DOWNLEFT,
                Action.UPFIRE,
                Action.RIGHTFIRE,
                Action.LEFTFIRE,
                Action.DOWNFIRE,
                Action.UPRIGHTFIRE,
                Action.UPLEFTFIRE,
                Action.DOWNRIGHTFIRE,
                Action.DOWNLEFTFIRE,
            ]

    def obs_to_flat_array(self, obs):
        flat, _ = ravel_pytree(obs)
        return flat.astype(jnp.float32)

    def _get_all_rewards(self, previous_state, state):
        if getattr(self, "reward_funcs", None) is None:
            return jnp.zeros(1)
        return jnp.array([rf(previous_state, state) for rf in self.reward_funcs])

    def step(self, state, action):
        out = _orig_step(self, state, action)
        if len(out) == 5:
            obs, new_state, reward, done, info = out
        else:
            # gymnasium-style 6-tuple from some forks
            obs, new_state, reward, terminated, truncated, info = out
            done = terminated
        all_rewards = _get_all_rewards(self, state, new_state)
        if hasattr(info, "_asdict"):
            d = dict(info._asdict())
        elif hasattr(info, "__dataclass_fields__"):
            d = {k: getattr(info, k) for k in info.__dataclass_fields__}
        elif isinstance(info, dict):
            d = dict(info)
        else:
            d = {
                "successful_rescues": getattr(new_state, "successful_rescues", 0),
                "difficulty": getattr(
                    getattr(new_state, "spawn_state", None), "difficulty", 0
                ),
                "step_counter": getattr(new_state, "step_counter", 0),
            }
        d["all_rewards"] = all_rewards
        return obs, new_state, reward, done, d

    JaxSeaquest.__init__ = __init__
    JaxSeaquest.obs_to_flat_array = obs_to_flat_array
    JaxSeaquest._get_all_rewards = _get_all_rewards
    JaxSeaquest.step = step
    JaxSeaquest._cbl_nexus_reward_funcs_patched = True
    print("Patched JaxSeaquest with NEXUS reward_funcs / obs_to_flat_array API")


def map_disable_enemies_wrapper_to_no_enemies_mod() -> None:
    """NEXUS create_env(no_enemies=True) → robustness ``no_enemies`` mod body."""
    try:
        import jax
        from jaxtari.wrappers import JaxatariWrapper
        import jaxtari.games.mods.seaquest_mods as sq_mods

        class DisableEnemiesWrapper(JaxatariWrapper):
            """Drop-in for NEXUS → robustness ``no_enemies`` mod."""

            @functools.partial(jax.jit, static_argnums=(0,))
            def step(self, state, action):
                obs, new_state, reward, done, info = self._env.step(state, action)
                new_state = new_state.replace(
                    shark_positions=jnp.zeros_like(new_state.shark_positions),
                    sub_positions=jnp.zeros_like(new_state.sub_positions),
                    enemy_missile_positions=jnp.zeros_like(
                        new_state.enemy_missile_positions
                    ),
                    surface_sub_position=jnp.zeros_like(new_state.surface_sub_position),
                )
                return obs, new_state, reward, done, info

        sq_mods.DisableEnemiesWrapper = DisableEnemiesWrapper
        print("Mapped NEXUS DisableEnemiesWrapper → robustness no_enemies mod")
    except Exception as e:
        print(f"WARNING: could not map DisableEnemiesWrapper to no_enemies: {e}")


def patch_atari_wrapper_kwargs() -> None:
    """Installed AtariWrapper lacks NEXUS noise kwargs (detect_prob/std_dev)."""
    import inspect
    from jaxtari.wrappers import AtariWrapper

    if getattr(AtariWrapper, "_cbl_nexus_kwargs_patched", False):
        return
    sig = inspect.signature(AtariWrapper.__init__)
    if "detect_prob" in sig.parameters:
        return
    _orig = AtariWrapper.__init__
    allowed = set(sig.parameters) - {"self"}

    def __init__(self, env, *args, **kwargs):
        kwargs.pop("detect_prob", None)
        kwargs.pop("std_dev", None)
        sticky = kwargs.get("sticky_actions", None)
        if isinstance(sticky, bool):
            kwargs["sticky_actions"] = 0.25 if sticky else 0.0
        # Drop anything else the fork doesn't know about.
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        # positional sticky/episodic may be in args — leave as-is
        return _orig(self, env, *args, **filtered)

    AtariWrapper.__init__ = __init__
    AtariWrapper._cbl_nexus_kwargs_patched = True
    print("Patched AtariWrapper to ignore NEXUS-only kwargs (detect_prob/std_dev)")


def patch_object_centric_defaults() -> None:
    """Modern OC defaults clip_reward=True; NEXUS Atari used clip_reward=False."""
    import inspect
    from jaxtari.wrappers import ObjectCentricWrapper

    if getattr(ObjectCentricWrapper, "_cbl_nexus_oc_patched", False):
        return
    sig = inspect.signature(ObjectCentricWrapper.__init__)
    if "clip_reward" not in sig.parameters:
        return
    _orig = ObjectCentricWrapper.__init__

    def __init__(self, env, *args, **kwargs):
        kwargs.setdefault("clip_reward", False)
        return _orig(self, env, *args, **kwargs)

    ObjectCentricWrapper.__init__ = __init__
    ObjectCentricWrapper._cbl_nexus_oc_patched = True
    print("Patched ObjectCentricWrapper default clip_reward=False (NEXUS parity)")


def patch_five_tuple_step_api() -> None:
    """NEXUS trainers unpack 5-tuples; modern wrappers return gymnasium 6-tuples.

    Collapse ``(obs, state, reward, terminated, truncated, info)`` →
    ``(obs, state, reward, done, info)`` with ``done = terminated | truncated``
    on the outermost MultiRewardLogWrapper (and PixelObs if used bare).
    """
    import jax
    from jaxtari.wrappers import MultiRewardLogWrapper

    if getattr(MultiRewardLogWrapper, "_cbl_nexus_five_tuple_patched", False):
        return

    _orig_step = MultiRewardLogWrapper.step

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        out = _orig_step(self, state, action)
        if len(out) == 5:
            return out
        obs, env_state, reward, terminated, truncated, info = out
        done = jnp.logical_or(terminated, truncated)
        return obs, env_state, reward, done, info

    MultiRewardLogWrapper.step = step
    MultiRewardLogWrapper._cbl_nexus_five_tuple_patched = True
    print("Patched MultiRewardLogWrapper.step → NEXUS 5-tuple (done=term|trunc)")


def prepare_nexus_runtime(
    *,
    nexus_root: Path,
    jaxatari_root: Path | None,
    use_vendored_seaquest: bool = False,
) -> None:
    """Configure imports for NEXUS train / mod-eval.

    Parameters
    ----------
    use_vendored_seaquest
        If True, use NEXUS's bundled JAXAtari (old Seaquest). Default False:
        use ``jaxatari_root`` / the installed robustness fork.
    """
    apply_nexus_source_fixes(nexus_root)
    prepend_sys_path([nexus_root / "src"])
    # CBL helpers (hrl_env, mod_eval) live under ConceptBasedLearning/scripts.
    cbl_scripts = Path(__file__).resolve().parents[1]
    prepend_sys_path([cbl_scripts])

    if use_vendored_seaquest:
        install_jaxatari(nexus_root / "JAXAtari" / "src", label="NEXUS vendored")
    else:
        if jaxatari_root is None:
            raise ValueError("jaxatari_root required unless use_vendored_seaquest=True")
        install_jaxatari(Path(jaxatari_root) / "src", label="installed / JAXATARI_ROOT")
        ensure_reward_funcs_api()
        patch_atari_wrapper_kwargs()
        patch_object_centric_defaults()
        patch_five_tuple_step_api()
        map_disable_enemies_wrapper_to_no_enemies_mod()
