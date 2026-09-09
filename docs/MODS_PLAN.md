# Robustness mods — parent categories

Date: 2026-09-08. Expands `EVAL_PLAN.md` §4, which left this as "sketch only —
own session later".

**Decisions locked this session:**
- Seven parent categories (§3), replacing the old `control` / `semantic` /
  `dynamics` / `visual` split for *reporting* purposes.
- Bucket E runs **both** blind and relabelled conditions.
- Bucket F (control / embodiment) is **in**, full bucket.
- The three S4 self-attack probes (`sparse_world`, `tiny_divers`, decoys) are **in**.
- Depth: **Seaquest gets the full grid** (~17 mods / 20 eval conditions); every
  other env gets a short 4–6 mod strip on A / B / C / G only.

Inputs to this draft:
- Locked-spine Seaquest RQ1 per-mod numbers
  (`outputs/models/paper_eval/eval/all_mod_eval_summary.csv`, mtime 2026-09-05).
- HackAtari (Delfosse et al., arXiv 2406.03997) taxonomy and its 50 mods / 16 games.
- What the ego observation actually encodes (`observations/egocentric.py`,
  `observations/skill_ego_features.py`, `environments/*/label.py`).

---

## 1. Why not just reuse HackAtari's taxonomy

HackAtari splits mods into **Visual domain adaptation / Dynamics adaptation /
Curriculum RL / Reward signal adaptation**. Two problems for us:

1. **It is a taxonomy of *how the mod was implemented*, not of *which agent
   assumption it breaks*.** "Dynamics" holds both `gravity` (physics) and
   `lazy_enemy` (a shortcut-learning probe) — those test completely different
   things. Our contribution is a specific *representation*, so our buckets have
   to name the representational assumption under test, otherwise a red cell in
   the table is not diagnostic.
2. **"Curriculum RL" is not an evaluation category at all** — it is a training
   use-case. In eval it just means "the mod made the game easier", and easier
   mods break the retention metric (see §4).

HackAtari's actual mod content is also thin for our purposes: per game it is
mostly `color_X`, `disable_X`, and a drift/gravity RAM poke. Our Seaquest fork
already has 27 mods, more than HackAtari has for any single game. So we are not
adopting their list; we are adopting their *reporting discipline* (agents vs
humans vs oracle on the same grid) and their two genuinely load-bearing ideas:
**shortcut exposure** (Pong `lazy_enemy`, Kangaroo punch-vs-joey) and
**misalignment exposure** (Seaquest infinite-oxygen farming).

---

## 2. What the ego observation actually assumes

The proposed categories are derived from this. The ego pipeline is five
separable stages, each with exactly one assumption we can attack:

| # | Stage | Where | Assumption |
|---|-------|-------|------------|
| S1 | Game dynamics | jaxatari env | — (this is the world) |
| S2 | Object → semantic label | `environments/<env>/label.py` | the hand-authored label map is **correct** |
| S3 | Target selection | labeling wrapper / meta | the **nearest active object** of the skill's class is the right one (single slot) |
| S4 | Ego encoding | `egocentric.py` | distance **clipped at 50px**, bearing quantised to **φ8** in a **facing-snapped** body frame, and **no size / no velocity / no identity** is enough |
| S5 | Action mapping | `observations/action_space.py` | the 18-action facing-frame strafe mapper matches the body |

S4 is the thesis. S2/S3 are the parts a reviewer will attack as "hand-engineered
oracle". S5 is the part that is invisible in the current mod suite.

---

## 3. Proposed parent categories

Seven buckets. Each names one assumption, has one predicted direction, and has
one named foil. Target 8–12 mods per env means roughly 1–2 per bucket.

### A. Appearance — *invariance claim*
Pixels change; geometry, dynamics and semantics are bit-identical.

- **Attacks:** nothing in our pipeline (by construction).
- **Predict:** ego retention **exactly 1.00**; PQN pixel drops hard.
- **Foil:** PQN pixel.
- **Evidence:** `random_color_enemies` — ego **1.00**, pqn/oc 1.00, pqn/pixel **0.27**.
- **Note:** because ego is *provably* invariant here, one mod is enough as a
  sanity/plumbing check, and it should be a *hard* one (full sprite swap +
  background clutter), not a palette poke. A second one is only worth it as a
  pixel-baseline stress. This is the cheapest column in the paper and also the
  least interesting; do not spend 4 mods on it.

### B. Layout — *relative-not-absolute claim*
Same objects, same rules, different spatial arrangement.

- **Attacks:** any memorisation of absolute screen coordinates.
- **Predict:** ego ≫ pixel; ego well above 1.0 relative to pixel even where both drop.
- **Evidence:** `shift_lanes` — ego 0.52 vs pqn/pixel **0.06** (8× better, both hurt).
  `inverted_lanes` — ego 0.94 vs pqn 1.57 (**ego loses**; worth understanding
  before we publish this bucket, likely a heuristic/meta scheduling artifact).
- **Seaquest:** `shift_lanes`, `inverted_lanes` (both exist).

### C. Kinematics — *honest-weakness bucket*
Speeds, accelerations, gravity, drift, oscillation, erratic motion.

- **Attacks:** S4's missing **velocity** channel.
- **Predict:** ego hurts, sometimes worse than pixel. We should say so.
- **Evidence:** `faster_enemies` — ego **0.00** (80 pts from a 22.6k base), the
  single largest ego collapse in the suite; pqn/pixel 0.08, heuristic 0.67.
  Meanwhile `gravity` ego 0.67 vs pqn 0.28 (ego wins — gravity is a *player*
  kinematics change, not a target one). That asymmetry is a genuinely good
  finding: ego is robust to *its own* dynamics changing and fragile to *target*
  dynamics changing, exactly as the feature set predicts.
- **Sub-split worth keeping:** player-kinematics vs target-kinematics.

### D. Object-set composition — *single-slot claim*
Add / remove / re-mix object types without changing per-type rules.

- **Attacks:** S3 nearest-target selection and skill scheduling.
- **Evidence:** `only_sharks` **1.85** vs `only_submarines` **0.18** — a 10×
  asymmetry from removing one of two DESTROY subtypes. `no_divers` ego 0.00
  while heuristic is 4.33. These are the most under-explained cells we have.
- **Best new mod in this bucket: decoys.** Objects that are geometrically and
  visually identical to a target but inert (or lethal), so nearest-target
  selection picks the wrong slot. This attacks the largest architectural
  simplification in the whole system and no existing benchmark has it.

### E. Semantic re-binding — *versatility showcase*
An object class changes meaning: collectible becomes hazard, hazard becomes
collectible, a new object type appears.

- **Attacks:** S2, the hand-authored label map.
- **Must be run as two conditions:**
  - **E1 "blind"** — label map unchanged. Measures how much the agent leans on
    a now-wrong prior. Expect collapse; report it honestly.
    Evidence: `lethal_divers` ego **0.03**.
  - **E2 "relabelled"** — label map edited to the new ground truth, **no
    retraining, no finetuning**. If ego skills recover most of base performance
    from a one-line label edit, that is a claim a pixel or end-to-end agent
    structurally cannot make. **This is the strongest single argument for the
    architecture and we currently do not run it.**
- **Cost is low:** the fork already implements `swap_diver_enemy_labels`,
  `collectable_sharks`, `extra_enemy_type`, `penalize_diver_shooting` — all four
  are in the JAXAtari REGISTRY but **missing from `mods/seaquest.yaml`**
  (still sitting under `planned:`). E2 needs only a label-map override in our
  repo, not a train.

### F. Control / embodiment — *mapper claim* (new; nothing tests this today)
Change the action↔world relation while leaving perception untouched: inverted
controls, mirrored world, added inertia/momentum, reduced action set,
rotated/8-way heading.

- **Attacks:** S5, the facing-frame POV mapper, in isolation from S4.
- **Why it matters:** the mapper is the least-defended part of the contribution
  and §2 of `EVAL_PLAN.md` already admits "ego LEFT means strafe, not rotate".
  A `mirror_world` mod is the cleanest possible test that ego is genuinely
  body-relative — a truly egocentric agent should be **exactly flat** under a
  horizontal flip, and a screen-space agent should collapse. That is an
  A-category-strength invariance claim on the *motor* side.
- **Bonus:** this bucket is also the honest bridge to §2's Time Pilot /
  Asteroids story (strafe ≠ turn). An `instant_turn` / momentum mod on Seaquest
  previews why SQ→Asteroids ZS is allowed to fail.
- **All new implementations.** Cheap ones: `mirror_world` (flip x in the
  state/renderer), `inverted_controls` (permute the action table),
  `momentum` (low-pass the player velocity).

### G. Objective re-specification — *alignment bucket*
Same world, different scoring: reward the other mechanic, penalise the
shortcut, zero out a score source.

- **Attacks:** whether score comes from the *labeled* mechanic or a shortcut.
- **Evidence, and it is damning for pixel:** on `no_enemies` / `disable_enemies`
  PQN pixel scores **203,547** vs ego **28,957**. PQN is farming a degenerate
  loop; ego stays near its base rate (1.28×). Same story on `peaceful_enemies`
  (pqn 219k, ego 127k). This is HackAtari's Q2 finding reproduced, and it is
  currently *hidden* in our report because these mods are excluded from the
  retention summary as "point farming". -> Note: 'unfortunately' this is a gotcha, 
  if there are no enemies nearby collecting divers and surfacing is the environment aligned thing to do. 
  So PQN is right here, the question is why ego failed here. Needs further investigation.
- **Fix:** stop treating this as contamination. Make it a designed bucket with
  a proper alignment metric (§4), not an exclusion list in `metrics.py`.
- `penalize_diver_shooting` already exists in the fork and is exactly this.

### Deliberately *not* a parent category: difficulty / CRL
HackAtari's third paradigm. Simplification confounds the metric rather than
testing a capability — see `slower_enemies` ego 4.18, `peaceful_enemies` 5.63.
Instead, tag every mod with a **direction** attribute (`harder` / `easier` /
`neutral`) and report easier mods against an oracle ceiling (§4). Keep 1–2
easier mods total, as the Delfosse-style "humans go up, PQN goes down" cell —
not as a whole bucket.

---

## 4. Metric problems this analysis exposed (blocking)

These matter as much as the category list; the current table is not publishable
as-is.

1. **Unbounded scores.** `disable_enemies` produces 200k–500k returns. Retention
   is then meaningless and the figure axis is dominated by one cell. Need a hard
   frame cap *and* score cap, or per-frame normalised score.
2. **Retention is a bare ratio.** Unbounded above, and undefined when the base is
   tiny — `skills/oc` base is 133 and `heuristic` base is 20, so their retention
   columns are noise (oc mean retention 1.578, heuristic 1.561, both fake).
   Needs a bounded, base-aware metric, e.g. normalising against a per-mod
   oracle/heuristic ceiling instead of against the agent's own base.
3. **No solvability control.** `dense_spawns` is **0.00 for every single arm
   including the heuristic**. That mod is broken, not a robustness result. Every
   mod must ship with an oracle/heuristic run proving it is solvable, and mods
   that fail that gate get dropped, not reported.
4. **Direction-blind reporting.** For an easier mod the question is "did you go
   up as much as the oracle went up", not "did you retain".
5. **Stale artifacts.** `paper_eval_report_1408.pdf` and its figures are from
   Aug 27 and predate the v2 ego (Aug 30) and the POV pixel/oc evals (Sep 1);
   several heatmap cells disagree with the current CSV. Also, the report builder
   groups by `agent_label` only and can pick the wrong ego stem alphabetically —
   it must filter on `config_stem`.
6. **Own-base retention is a 1/base ranking, not robustness.** On the
   surfacesub5 clip sweep, several load-bearing mods sit in a *shared absolute
   band* across clip10–clip50 (e.g. `gravity` ~15–19k, `shift_lanes` ~20k,
   `slow_oxygen_drain` ~16–18k). The agent's **base** still moves a lot
   (clip10 18.6k vs clip50 35.3k), so `mod_mean / agent_base` then punishes
   the stronger in-domain agent. Clip50's dynamics median 0.46 vs clip10's 0.97
   is mostly that artifact, not "clip50 is less robust." Nested `distflags` /
   overlapping `distbins` have the same problem plus they reconstruct Seaquest
   metres (see `tmp/ego_min_surfacesub5_distabl/`).

   **Do not** rank ego vs PQN, or clip arms against each other, by category
   median own-base retention.

   **Do** for the near-term ego vs PQN story:
   - Report **absolute** returns on a small stress set that actually moves:
     `faster_enemies`, `fast_oxygen_drain`, `surface_sub_always`, `gravity`,
     `shift_lanes`.
   - Keep `random_color_enemies` as the appearance claim (ego 1.00 vs PQN
     pixel 0.27) — that ratio is valid because the world is bit-identical.
   - Tag farming / easier mods (`disable_enemies`, `no_enemies`,
     `slower_enemies`, `peaceful_enemies`) as **ceilings / alignment**, not
     retention. Cap or drop them from any aggregated "robustness" number
     (items 1, 4, 11).
   - Drop `dense_spawns` (item 3).
   - Until an oracle exists, compare ego and PQN **on the same absolute
     cells**, not each against its own base. PQN's higher own-base retention
     on some dynamics mods has already been a high-base / low-competence
     mirage relative to ego's mid-band scores.

   Exclusive near/mid/far distance (`EGO_DISTANCE_HARD_EDGES_PX: [12, 40]`)
   is the next encoding probe so the S4 "bounded range" claim is a few
   symbols, not a magic clip. Sticky (Machado 0.25 at train) is a *separate*
   clip50 arm — do not confound it with the encoding. That does not replace
   the metric fix above.

---

## 5. Seaquest — the full grid (locked)

`E` = already in the fork registry, `N` = needs implementing.
`dir` = direction attribute (§4.4). 17 mods, 20 eval conditions incl. base.

| Bucket | key | dir | Status | What it breaks / expected |
|---|---|---|---|---|
| — | `base` | — | E | control |
| A | `random_color_enemies` | neutral | E | ego exactly 1.00; pixel 0.27 (known) |
| A | `mines` | neutral | E | sprite swap, harder than palette poke |
| B | `shift_lanes` | harder | E | absolute-position memorisation; ego 0.52 vs pixel 0.06 |
| B | `inverted_lanes` | harder | E | ego 0.94 vs pixel 1.57 — **diagnose before publishing** |
| C | `faster_enemies` | harder | E | missing velocity (target); ego 0.00 — headline weakness |
| C | `gravity` | harder | E | player kinematics; ego 0.67 vs pixel 0.28 — ego wins |
| D | `only_submarines` | harder | E | DESTROY subtype removal; ego 0.18 vs `only_sharks` 1.85 |
| D | `mirror_divers` | harder | **N** | **decoys** — inert diver-shaped objects break nearest-target |
| E1 | `lethal_divers` | harder | E | blind: label map wrong; ego 0.03 |
| E2 | `lethal_divers@relabel` | harder | E + override | **recovery without retraining** |
| E1 | `collectable_sharks` | neutral | E (modpack) | blind |
| E2 | `collectable_sharks@relabel` | neutral | E + override | **recovery without retraining** |
| F | `mirror_world` | neutral | **N** | horizontal flip — ego should be **exactly flat** |
| F | `inverted_controls` | harder | **N** | action-table permutation; mapper only |
| F | `momentum` | harder | **N** | player inertia; previews strafe≠turn (Asteroids) |
| G | `penalize_diver_shooting` | harder | E | scores via labeled mechanic or shortcut? |
| G | `no_enemies` | easier | E | alignment: pixel 203k vs ego 29k — **needs cap** |
| S4 | `sparse_world` | harder | **N** | targets kept >50px → distance channel saturates |
| S4 | `tiny_divers` | harder | **N** | shrink hitboxes → sub-φ8-bin aim precision |

Dropped: `dense_spawns` (0.00 for every arm including heuristic — broken, fails
the solvability gate in §4.3).

`paper_core` subset if the full grid does not fit a figure: `base`,
`random_color_enemies`, `shift_lanes`, `faster_enemies`, `gravity`,
`only_submarines`, `mirror_divers`, both `lethal_divers` conditions,
`mirror_world`, `no_enemies`, `sparse_world`.

### Which claims each bucket carries

- **A + F(`mirror_world`)** — the two *invariance* cells. Ego should be exactly
  1.00 on both: one on the perceptual side, one on the motor side. Pixel should
  collapse on both. This pairing is the cleanest slide in the paper.
- **C + S4** — the honest weaknesses. Say them plainly.
- **E2** — the versatility claim, and the only cell no baseline can even attempt.
- **G** — the alignment claim, reproducing HackAtari's Q2 with a better foil.

## 6. Other environments — short strips (A / B / C / G only)

4–6 mods each, no bucket D/E/F, no S4 probes. These exist to show breadth, not
to re-run the Seaquest analysis.

| Env | A | B | C | G |
|---|---|---|---|---|
| Pong | `change_player_color` | `shift_player` | `linear_movement` | **`lazy_enemy`** (headline shortcut probe) |
| Breakout | `block_color` | `small_paddle` | `ball_drift` | — |
| Phoenix | `grayscale` | **N** formation reshuffle | `fast_enemy_bullets` | — |
| Space Invaders | — | `shift_shields` | — | `disable_shields` |
| Beamrider | `toaster` | `three_lanes` | `hardcore` | `mothership_laser` |
| Freeway | `bikes` | — | `slow_cars`, `invert_speed` | — |

Notes:
- Phoenix's 11 fork mods are almost all appearance (`night_mode`, `grayscale`,
  `inverted_colors`, `matrix_theme`, `blood_moon`) plus player buffs. **A is
  over-covered; B is empty.** One new formation-reshuffle mod fixes that.
- Freeway stays deliberately thin — it is the limitation discussion, not a grid.

### Cost warning for `EVAL_PLAN.md` §1

`EVAL_PLAN.md` promotes **Name This Game** to a base and possibly **Time Pilot**.
Both exist as envs but have **zero mods** in the fork (no `MOD_MODULES` entry).
Kaboom is deferred (not in upstream jaxatari yet; label/policy code is parked
unregistered). So the workshop implicitly includes writing a mod registry from
scratch for NTG (and TP if promoted) — budget it now.

## 7. Work list

**JAXAtari fork — 6 new Seaquest mods** (plugin class + one `REGISTRY` line each):

1. `mirror_divers` — D. Reuse spare enemy slots, diver sprite/hitbox, no pickup
   effect. No new entity type needed (see §8 limit).
2. `mirror_world` — F. Flip x in the post-step state and in the renderer.
   Verify the flip is total: player, targets, scroll direction, sprites.
3. `inverted_controls` — F. Permute the action table in `_player_step`.
4. `momentum` — F. Low-pass the player velocity in `_player_step`.
5. `sparse_world` — S4. Constrain spawn x/y so the nearest target is usually
   beyond 50px. Must not simply reduce density to the point of trivialising.
6. `tiny_divers` — S4. Shrink diver hitboxes via `constants_overrides`.

Plus **1 new Phoenix mod** (formation reshuffle, bucket B).

**Main repo:**

7. Add the four already-implemented-but-unwired Seaquest mods to
   `configs/experiment/paper_eval/mods/seaquest.yaml`: `swap_diver_enemy_labels`,
   `collectable_sharks`, `extra_enemy_type`, `penalize_diver_shooting` (currently
   stranded under `planned:`).
8. **Label-override mechanism for E2.** Precedent exists —
   `set_seaquest_surface_label` in `environments/seaquest/label.py` already does
   exactly this for one object class. Generalise it to a per-env override map
   driven from config, then add a `relabel:` key to the mod YAML schema.
9. Add `direction: harder|easier|neutral` to the mod YAML schema and to the
   summary CSV.
10. Rewrite the reporting buckets from `control/semantic/dynamics/visual` to the
    seven categories. Keep the old field for backwards compatibility with
    existing CSVs.

**Metrics (blocking — see §4):**

11. Hard frame **and** score cap so `no_enemies` cannot emit 200k–500k returns.
12. Replace bare retention with an oracle-normalised, bounded metric; drop the
    `skills/oc` and `heuristic` retention columns or mark them invalid
    (bases of 133 and 20). Until then, **do not rank agents by own-base
    dynamics median** — report absolute stress-set scores and appearance
    invariance separately (§4.6).
13. Solvability gate: every mod ships with a heuristic/oracle run; mods where
    the oracle also scores ~0 get dropped, not reported.
14. Regenerate the report from the current `eval/` and fix the builder to filter
    on `config_stem`, not `agent_label`.

---

## 8. Implementation notes (from the fork)

Writing new mods is cheap. Two plugin bases in
`src/jaxatari/modification.py`:

- `JaxAtariInternalModPlugin` — monkey-patch any env/renderer method by name,
  plus `constants_overrides`, `asset_overrides`, `attribute_overrides`.
- `JaxAtariPostStepModPlugin` — `run(prev_state, new_state)` state surgery; the
  wrapper then recomputes obs/reward/done/info, so **reward mods work**.

Registration is one line in the game's `REGISTRY` dict. Mods **stack**
(`mods: [a, b]` in our YAML, and train-time `GAME_MODS` are unioned onto every
eval condition by `merge_train_and_eval_game_mods`).

Two real limits:
- **No first-class new entity type** with new state fields. So `mirror_divers`
  and `extra_enemy_type` have to reuse existing object slots.
- **No action-space resizing.** F-bucket mods must permute or constrain the
  existing action table, not extend it.

Our YAML already supports what we need: `key`, `mods`, `category`, `skip_eval`,
`paper_core`. Adding `direction: harder|easier|neutral` and
`relabel: true|false` (for E2) would cover the new schema needs.
