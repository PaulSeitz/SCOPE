# POV frame backlog

> **After the current paper eval run finishes**, fix the remaining broken `_MOVE_SPECS`
> tables before trusting transfer results on these envs.
>
> Verify with: `python scripts/pov_frame_playground.py --env <name>`
>
> Full sweep: `JAX_PLATFORMS=cpu python scripts/pov_frame_playground.py --all --out tmp/pov_playground`

## Fixed (2026-08-30)

- **Obs/mapper facing rotation** — `resolve_facing_deg` now snaps to per-env view axes;
  sprite orientation decoded with JAXAtari convention (0°=up → screen −90° offset).
- **Pong `move_fire_table`** — `FORWARD_F`/`BACKWARD_F` were `LEFTFIRE` (paddle down);
  now `FIRE` into the field; lateral fire uses `RIGHTFIRE`/`LEFTFIRE`.

## Open — low priority until post-eval

| Env | Symptom | Likely fix |
|-----|---------|------------|
| **tennis** | 1 probe moves AWAY; table is Pong's 1-D paddle spec | Full 8-way table from measured `ACTION_SET` displacement |
| **skiing** | `BACK_*` ego actions move AWAY; only LEFT/RIGHT in `ACTION_SET` | Custom 2-D strafe table (like Freeway) |
| **choppercommand** | 3 probes move AWAY; ALE RIGHT displaces left | Per-env table from measured movement, not Seaquest copy |
| **spaceinvaders** | `BACKWARD` resolves to `RIGHTFIRE`; indices 6–9 OOB | 6-action `ACTION_SET` table: `[NOOP,FIRE,RIGHT,LEFT,RIGHTFIRE,LEFTFIRE]` |

Code markers: search `TODO(POV_BACKLOG)` in `src/concept_based_learning/observations/pov_relative.py`.

## Retrain required

All **facing-frame Seaquest skill checkpoints** trained before the obs/mapper fix are
invalid (obs frame was rotated +90° relative to actions). Use the `_v2` configs and
`pov_facing_v2*` suites in `scripts/run_paper_eval.py` / `scripts/run_pov_facing_v2_retrain.sh`.
