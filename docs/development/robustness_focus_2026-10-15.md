# Focused robustness candidates — 2026-10-15

## TODO (after thesis_final paper_core sweep)

**Run these `*_robustness_focus` cells on frozen thesis checkpoints.** Several
current paper_core mods (esp. outside Seaquest) do not cleanly test the
layout-shift vs task-preserving-simplification distinction we want; the probes
below were added for that reason and are *not* wired into `thesis_final` yet.

- Allowlists: `configs/experiment/paper_eval/mods/<game>_robustness_focus.yaml`
- Fork: `JAXAtari-seaquest-robustness` (`robustness_probes.py` + registry keys)
- How: set `PAPER_EVAL_MODS: <game>_robustness_focus` on eval-only configs
  reusing existing `tf_*` checkpoints (see “Selection and reproducibility”).
- Also noted in `docs/thesis_final_RUNLOG.md` → “Follow-up”.

The thesis takes precedence over older MODS_PLAN categories. The focus is now
(1) modest shifts to previously fixed coordinates and (2) task-preserving
simplifications / alignment checks. Appearance and semantic-role demonstrations
can remain Seaquest-only. Seaquest mods are unchanged.

These are implemented candidates, not evidence of agent robustness or failure.
No trained-checkpoint evaluation was launched. Mild coordinate shifts still
need a competent control to establish that difficulty has not risen materially.

## Implemented candidates

Each row contains TWO SEPARATE evaluation conditions, never a modpack.

| Environment | Layout key and change | Simplification key and change |
|---|---|---|
| Name This Game | `shift_tentacle_columns`: all eight tentacle bases +8px X | `infinite_oxygen`: keep oxygen full; all enemies, scoring and wave progression retained |
| Time Pilot | `shift_player_anchor`: fixed player X 76 -> 84, with native shot/collision coordinates | `no_enemy_fire`: remove enemy missiles, retain aircraft and scoring |
| Space Invaders | `shift_shield_row`: bunkers Y 157 -> 151 | `no_enemy_fire`: remove alien bullets, retain aliens, UFO and bunkers |
| Beamrider | `shift_player_row`: player Y 164 -> 160; seven lanes retained | `no_enemy_shots`: remove UFO shots, retain UFO movement/patterns and other hazards |
| Asteroids | `shift_initial_rocks`: initial active rocks +8px Y, native later spawns | `no_ship_collisions`: remove ship-rock collision damage, retain rock motion, missile kills and splitting |
| Pong | `shift_player_plane`: player X 140 -> 132 | `slower_enemy`: opponent step 2 -> 1 with native tracking; existing `lazy_enemy` remains a useful complementary probe |
| Breakout | `shift_paddle_row`: paddle Y 189 -> 181 | `wider_paddle`: normal 16 -> 24px, shrunken 12 -> 20px; retain shrinking and speed tiers |
| Freeway | `shift_car_rows`: each car row +3px Y | `no_cars`: remove traffic from collision processing, rendering and OC activity; retain crossing score |
| Phoenix | `shift_player_row`: player Y 173 -> 169 | `no_enemy_fire`: suppress firing in bird, bat and boss waves; retain enemies and movement |

Implementation: JAXAtari `src/jaxtari/games/mods/robustness_probes.py`,
registered through each game's existing controller. Geometry uses constants
where possible so rendering, collision and object observation agree. Initial
Asteroids placement uses a reset-only hook with a recomputed observation.
Enemy-fire removal occurs before collisions, not through post-hoc life repair.

`no_ship_collisions` leaves dangerous-looking rocks visible deliberately: it
removes the survival cost while keeping the shooting task. It is a hazard
neutralization probe, not object removal. `wider_paddle` expands collision
geometry and changes width-dependent interception/bounce geometry; classify it
as a simplification, not a pure unchanged-mechanics layout test. Likewise,
`slower_enemy` changes opponent dynamics. The intended distinction is the
capability tested, rather than pretending simplification is a disjoint MDP axis.

## Selection and reproducibility

Dedicated allowlists live at
`configs/experiment/paper_eval/mods/<game>_robustness_focus.yaml`. Each selects
base plus the two independent candidates, including through `paper_core`.
The original per-game allowlists and generated thesis-final configs are unchanged.

For a new evaluation config using the existing trained checkpoint, set e.g.:

```yaml
ENV_NAME: timepilot
PAPER_EVAL_MODS: timepilot_robustness_focus
PAPER_EVAL_MOD_KEYS: [base, shift_player_anchor, no_enemy_fire]
```

Keep the existing architecture/checkpoint fields and evaluation seed/horizon
settings. These are config keys accepted by `scripts/run_paper_eval.py`, not
new command-line switches. The programmatic harness also accepts
`mods_allowlist="timepilot_robustness_focus"` and `mod_keys=[...]`.
Ensure the Seaquest-robustness checkout is first on PYTHONPATH.

## What the results can establish

- Compare each frozen policy to its own base, with identical evaluation
  horizons, exploration and seed protocol. Also show absolute scores.
- For simplifications, check that a competent control can exploit the easier
  condition. Improved optimal opportunity does not imply every fixed policy
  must improve, nor does a lower return alone prove shortcut learning.
- Inspect task progress and behavior: kills, waves/sectors, crossings, rallies,
  oxygen detours and time spent chasing absent threats. Longer survival can
  raise total score without improving task execution; report score rate too.
- Keep labels, heuristics and action maps frozen for the main robustness test.
  A selector that fails after threat removal is a complete-system failure.
  Attribute it to skills only after inspecting selected targets and skill
  behavior. Do not silently fix heuristics in the main condition.
- Compare native competent policies and frozen transferred skills separately.
  A policy with negligible base competence supplies little evidence here.
- Do not interpret traffic frozen across a lane as an assured Freeway
  simplification: it can become a permanent obstacle. Use true `no_cars`.
- Avoid stationary one-target farming rewrites as the headline check: those
  also change target count, movement, spawning and sometimes shooting.

For more controlled evidence of information restriction, pair layout returns
with skill-level states that keep the selected player-target pair fixed while
changing only other objects. Moving a player row tests sensitivity to geometry
and framing, but does not by itself isolate excluding surrounding objects.

## Useful follow-ups

- NTG: vary oxygen-line spawn X within the existing reachable boat range,
  without changing the line lifetime or refill rules.
- Phoenix: shift the ordinary formation by a small offset, preserving actor
  identity and dive rules; use this alongside the player-row shift.
- Freeway: vary initial car X phase with unchanged cadence timers.
- Seaquest: retain its established no-enemies and lane-shift demonstrations.

Behavioral checks live in JAXAtari `tests/test_robustness_probes.py`.

Breakout rendering now masks custom paddles to their active normal/shrunken
collision width. This keeps the wider-paddle probe consistent for pixels and OC.
The historical native-size sprite path is retained for checkpoint comparability;
it already draws the full normal sprite after native shrinking. Thus the new
wide-paddle images match their hitboxes, but native shrunken-paddle images retain
that known legacy discrepancy. Do not attribute this rendering issue to learning.
CBL integration checks live in `tests/test_robustness_focus.py`.
