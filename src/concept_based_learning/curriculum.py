from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SkillPhase:
    name: str
    game_mods: Optional[List[str]]  # None => do not set GAME_MODS (base env)
    total_timesteps: int
    step_offset_updates: int


def _split_timesteps_evenly(total_timesteps: int, n_phases: int) -> List[int]:
    if n_phases <= 0:
        return []
    if n_phases == 1:
        return [int(total_timesteps)]
    base = int(total_timesteps) // int(n_phases)
    rem = int(total_timesteps) % int(n_phases)
    return [base + (1 if i < rem else 0) for i in range(n_phases)]


def build_skill_phases(
    *,
    base_config: Dict[str, Any],
    total_timesteps: int,
    env_key: str,
    resolve_curriculum_mods,
) -> List[SkillPhase]:
    """
    Pure curriculum planner for skill training.

    - If CURRICULUM_ENABLED is false: one phase ('full') using base env (no GAME_MODS override).
    - If enabled: phases come from CURRICULUM_PHASES, and mods are resolved via
      `resolve_curriculum_mods(env_key, phase_name, CURRICULUM_MODS)`.
    - Timesteps are split evenly across phases; remainder goes to early phases.
    - `step_offset_updates` is computed as cumulative updates to keep logging steps continuous.
    """
    curriculum_enabled = bool(base_config.get("CURRICULUM_ENABLED", False))
    if not curriculum_enabled:
        return [
            SkillPhase(
                name="full",
                game_mods=None,
                total_timesteps=int(total_timesteps),
                step_offset_updates=0,
            )
        ]

    phase_names = list(base_config.get("CURRICULUM_PHASES", ["collect", "destroy", "avoid"]))
    curriculum_mods = base_config.get("CURRICULUM_MODS", {}) or {}
    phase_timesteps = _split_timesteps_evenly(int(total_timesteps), len(phase_names))

    num_steps = int(base_config.get("NUM_STEPS", 128))
    num_envs = int(base_config.get("NUM_ENVS", 512))
    update_divisor = max(1, num_steps * num_envs)

    phases: List[SkillPhase] = []
    offset_updates = 0
    for phase_name, phase_steps in zip(phase_names, phase_timesteps):
        mods = resolve_curriculum_mods(env_key, str(phase_name), curriculum_mods)
        phases.append(
            SkillPhase(
                name=str(phase_name),
                game_mods=list(mods) if mods else None,
                total_timesteps=int(phase_steps),
                step_offset_updates=int(offset_updates),
            )
        )
        offset_updates += int(phase_steps) // update_divisor

    return phases

