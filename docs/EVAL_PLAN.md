# Evaluation plan (locked decisions)

Date: 2026-09-07. Thesis claim: **strict informational abstraction** on System 1
(egocentric, bounded pose) yields in-domain competence, robustness, and
*homologous* skill transfer. Geometry / pose rewrites after φ8+facing+clip50
are frozen. Remaining work is (1) a few **channel-deletion** trains, (2) a
wider but shallower ALE matrix, (3) Montezuma as planned-env showcase,
(4) Freeway as a limitation, (5) a later dedicated mods workshop.

## Why this direction

- RQ1 on Seaquest is already showable (ego ≈ PQN pixel; pixel outline is
  load-bearing; OC skills failed in-domain).
- RQ2 is **unstable on non-homologous COLLECT** (Pong intercept ≠
  approach-to-contact). Homologous DESTROY already transfers (Beamrider,
  Phoenix). Extra AABB / coarse-pose arms made SQ and Pong worse.
- Simpler input is the thesis, but “simpler” means **delete extra channels**,
  not retune angle, POV, or bins. LCR×near/far was too coarse (source
  collapse). Bodygaps was too clever (source + transfer collapse).
- Depth comes from **more base environments**, not from over-fitting Seaquest
  geometry. Breadth comes from a **cross-transfer grid** (cheap ZS) plus
  scratch/FT ceilings.

---

## 0. Spine freeze + last ego-minimization runs

**Do not rework:** facing-locked POV mapper, facing-frame φ, clip50, φ8 bins,
v2 / navigatesurface *labeling* (surface ≠ COLLECT). Size, velocity, shot
geometry stay off.

**Do (few runs, v2 checkpoint recipe) — suite `ego_min_channels`:**

```bash
python scripts/run_paper_eval.py --suite ego_min_channels --gpus 0,1 --wait-deps --lean
```

Five 25M arms (trains first: GPU0×3, GPU1×2 → SQ full eval → Pong /
Beamrider / Phoenix ZS):

1. Drop blocking (`USE_EGO_BLOCKING_FEATURES: false`)
2. Drop blocking + `facing_enemy` (`USE_EGO_FACING_ENEMY_FEATURES: false`)
3. Drop blocking + all weapon extras (`USE_EGO_WEAPON_FEATURES: false`)
4. Drop `facing_enemy` only (blocking on)
5. Drop all weapon extras only (blocking on)

Gate: Seaquest base still in the ~20k band; Beamrider / Phoenix ZS not
collapsed vs v2 reference. Pong is a secondary readout, not the lock
criterion. Then **lock the ego stack** for the matrix below.

Optional follow-up: `EGO_LEGAL_ACTION_MASK: true` **at train** on the
winning minimal ego.

Quirk (not a planned arm): Pong labeled DESTROY instead of COLLECT
(“align with this object”). At most a discussion aside.

---

## 1. Environment roles

### Bases (skill train from scratch + heuristic)

| Env | Why |
|-----|-----|
| **Seaquest** | Main robustness env; **all** skill input modes (ego / pixel / oc). |
| **Phoenix** | Second shooter layout; labels already exist; Space Invaders is a free homologous target. |
| **Name This Game** | Homologous underwater world (COLLECT + DESTROY). Policy: when the catcher/treasure field appears, **stick COLLECT** and keep it selected while it is active; dense reward for staying inside. Easy heuristic. |

Time Pilot and Asteroids: **wire + smoke both**. At least one becomes a
**base** even if SQ→them fails (strafe ≠ turn family; see §2). Scratch +
transfer *between* them is the interesting cell.

### Transfer targets (same list from every base)

| Env | Role |
|-----|------|
| Space Invaders | Phoenix-family DESTROY |
| Beamrider | already a win; keep |
| Time Pilot | omni DESTROY, 8-way facing, no Chopper turbo |
| Asteroids | continuous facing; native POV test |
| Pong + Breakout | intercept family (drop Tennis) |
| Kaboom | cleaner 1D catch / lateral-null |
| Freeway | limitation (single-target vs gap) |
| Name This Game / Phoenix / Seaquest | cross-base (when not the source) |

**Out:** Galaxian / Assault / Air Raid (redundant with SI+Phoenix). Tennis.
Asterix (crowded COLLECT, expensive scratch, weak ZS). Chopper Command is
**smoke only** (scroll acceleration); not a headline cell.

### Showcases (not the transfer grid)

- **Montezuma room 1:** symbolic plan × frozen skills (scratch-on-env OK).
  Control: nearest-object heuristic + PQN scratch.
- **Freeway:** ZS vs native scratch/FT + discussion (doesn’t need to hit cars
  to still fail: no multi-car gap). No extra Freeway skill campaign.

---

## 2. Asteroids / Time Pilot POV (corrected)

The facing-frame **math** is `cos/sin(facing_deg)`, but
`_player_orientation_deg` **snaps** to the env’s declared view axes. Seaquest
is two axes (0° / 180°, L/R hull). Phoenix/Pong/SI are one axis (look into
the field). The mapper is the same: **two 18-action tables**, not a continuous
heading. φ8 is target bearing *in that snapped body frame* (full ±π, so the
target can be behind). It is not a 180° camera frustum.

So Seaquest is not “continuous facing that happens to look L/R.” It is
**strafe-in-8-world-dirs + flip the hull**. Ego LEFT means translate
perpendicular to facing, not rotate the ship.

- **Time Pilot (8-way):** extend the snap to 8 headings and 8 mapper tables.
  Still not continuous; still “pick a compass dir,” but more bins. LEFT will
  still mean *strafe* unless we change the mapper.
- **Asteroids:** true continuous heading. Honest mapping is FORWARD=thrust,
  LEFT/RIGHT=**turn** (no strafe). That is a different motor primitive than
  Seaquest. Existing `USE_POV_TURN_PRIMITIVES` is exactly this (6 actions,
  integrated θ_h) and is currently **incompatible** with the facing-frame
  flag — wiring Asteroids means choosing turn-style mapping, not pretending
  18-way strafe exists.

**Story we want:** SQ → TP/Asteroids ZS is allowed to fail (strafe ≠ turn).
Scratch-train one of them as a **base**, then transfer to the other. If that
works, skills learned the rotate-and-thrust mechanic; they just were not
born with it on Seaquest. That is homologous transfer inside a family, not
a failed Seaquest transfer.

---

## 3. Run matrix

Budget: **20M** skill/PQN trains (down from ~25M), **128 envs** unchanged,
**5M** finetune. Match PQN step count to skills. 10M is OK for transfer
*scratch* if 20M does not fit; do not mix 10M and 20M inside one comparison
column.

Cross-transfer ZS is **base env only** (no full mod grid per cell). A few
hours of eval, not a few days of train.

| Block | Where | Modes | Notes |
|-------|--------|-------|--------|
| Skill train | Seaquest | ego, pixel, oc | thesis: ego is better **here** |
| Skill train | Phoenix, NTG (+ TP/Asteroids if promoted) | **ego only** | universal finding, not a second pixel/oc bake-off |
| PQN baselines | all **bases** | pixel (+ oc if cheap) | required |
| PQN baselines | **transfer** envs | pixel, once | ceiling; one-time at the chosen step count |
| Scratch skills | all transfer envs | **ego** | pain point; 10–20M |
| Finetune 5M | every base → every transfer | ego | necessary ceiling vs ZS |
| Finetune 5M | Seaquest → transfers | pixel / oc as well | “full modes only on SQ” |
| RQ3 tandem | bases (SQ required; Phoenix/NTG nice) | pixel or oc meta + **frozen or jointly trained ego skills** | **no meta-only** |
| RQ3 eval | that env’s **full mod suite** | tandem | compositionality + robustness |
| RQ3 transfer | skills from tandem ckpt only | ego ZS + target heuristic | meta stays home; same grid as skills-only, to show tandem did not wreck System 1 |

**Minimal-ego ablation** (block 0) stays **outside** this matrix until the
spine is locked; do not spawn 3× pixel/oc copies of every strip.

---

## 4. Mods (sketch only — own session later)

Reuse Seaquest’s existing split (`control` / `semantic` / `dynamics` /
`visual` in `configs/experiment/paper_eval/mods/seaquest.yaml`) but **report
by intent**, not by the old three buckets. Target ~8–12 mods per env, not
SQ’s full list everywhere.

| Intent | ~count | Seaquest examples | What it should show |
|--------|--------|-------------------|---------------------|
| Simpler | 1–2 | `slower_enemies`, `peaceful_enemies`, `unlimited_oxygen` | humans/ego up; PQN often **down** (Delfosse-style) |
| Target / farming | 1–2 | `disable_enemies` / `no_enemies`, `no_divers` | alignment: score from the labeled mechanic, not a shortcut |
| Visual | 1–2 | `random_color_enemies` | OC/ego flat; PQN pixel drops |
| Harder dynamics | 1–2 | `faster_enemies`, `gravity` | ego may hurt too; honest weakness |
| Layout | 1–2 | `shift_lanes`, `inverted_lanes`; Phoenix formation | structure change without new sprites |
| New object rules | 1–2 | planned `collectable_sharks`, `extra_enemy_type`; Beamrider `mothership_laser` | semantic OOD |

PQN pixel is the visual + shortcut foil. Ego should win visual and (we
hope) simpler/farming; layout/new-rules are the stress columns.

Phoenix/NTG/Pong/Kaboom/SI get the **same intent grid**, implemented with
whatever jaxatari already has, then a short implement-list. Do not clone
Seaquest mods that do not exist (`shift_lanes` on Pong is already
`shift_player`). Full catalogue = later brainstorm; this table is the
template.

---

## 5. Order of work

1. **Minimal ego** trains + SQ/Pong/BR/PH eval → lock spine.
2. **Labels + heuristics:** NTG (sticky COLLECT), Time Pilot, Asteroids;
   Kaboom; Breakout already exists; SI exists.
3. **Smoke** TP / Asteroids / Kaboom / Chopper with frozen SQ ego (current
   spine). Decide TP/Asteroids base vs transfer-only.
4. **Mods workshop** (categories above × env). Implement missing jaxatari
   mods only for the locked ~8–12.
5. **Base trains** (SQ all modes, other bases ego + PQN) at 20M.
6. **Cross-ZS grid** (every base → every target, ego, base mod only).
7. **Scratch + FT** on transfers (ego; SQ also pixel/oc FT).
8. **RQ3 tandem** on bases + full mod eval; ZS the **skills** (not the meta).
9. **Montezuma L1** plan video + heuristic/PQN controls.
10. Write with Freeway as limitation, Pong/Breakout/Kaboom as intercept
    family, homologous DESTROY as the transfer win.

## Explicit non-goals

- More AABB / bin / clip / LCR pose arms.
- Multi-object attention / dual-slot revival.
- Kangaroo as flagship (timing). Keep as leftover if L1 ever clears.
- RQ3 meta-only and transferring a learned meta.
- Full pixel/oc bake-off on every ALE title.
