# Evaluation plan (locked decisions)

Date: 2026-09-07 (distance-clip section amended 2026-09-10; sparse noclip
lock 2026-09-16). Thesis claim:
**strict informational abstraction** on System 1 (egocentric, bounded pose)
yields in-domain competence, robustness, and *homologous* skill transfer.
Facing-frame φ8 + surfacesub5 labeling are frozen. **Distance clip locked
to noclip** under sparse C/D/A (see §0.1; clip10 remains the near-field
control). Remaining work:
(1) suite **`noclip_sparse_smoke`** — 20M ego scratch + flat PQN +
from-scratch tandem on bases/transfers (see §5),
(2) cross-base ZS / FT grid, (3) Montezuma showcase (deferred; policy
broken), (4) Freeway limitation, (5) mods workshop (`docs/MODS_PLAN.md`).
**Pong:** keep as intercept fail cell; no fix campaign.

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

**Do not rework:** facing-locked POV mapper, facing-frame φ, φ8 bins,
surfacesub5 / navigatesurface *labeling* (surface ≠ COLLECT). Size, velocity,
shot geometry stay off. Distance clip: **locked to noclip** under sparse C/D/A
(§0.1); do not reopen clip sweeps.

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

### Results (synced 2026-09-08) — `tmp/ego_min_channels/`

| Arm | SQ base | BR ZS | PH ZS | Pong | Gate |
|-----|---------|-------|-------|------|------|
| **v2 (ref)** | 22 600 | 788 | 3 460 | **+1** (21–20) | — |
| noblocking | 22 550 | 543 | **453** | −21 | fail (BR/PH) |
| **noblocking_nofacing** | **34 203** | **814** | **4 557** | −11 | **PASS (lock)** |
| noblocking_noweapon | **13 167** | 884 | 13 483 | +20 | fail (SQ) |
| nofacing | 28 697 | 603 | 6 463 | −10 | PASS |
| noweapon | 23 643 | 543 | 3 090 | **+4** | fail (BR) |

**Lock for the matrix:** `noblocking_nofacing` —
`USE_EGO_BLOCKING_FEATURES: false`, `USE_EGO_WEAPON_FEATURES: true`,
`USE_EGO_FACING_ENEMY_FEATURES: false` (keep `can_fire` only). Best
homologous DESTROY retention with SQ above v2. Pong regresses vs v2; do
not chase with more channels. Skip legal-mask train unless a later Pong
probe needs it. Ckpt:
`outputs/models/paper_eval/rq1/seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_nofacing/`.

Optional follow-up: `EGO_LEGAL_ACTION_MASK: true` **at train** on the
locked minimal ego (deferred).

Quirk (not a planned arm): Pong labeled DESTROY instead of COLLECT
(“align with this object”). At most a discussion aside.

---

## 0.1 Distance clip — surfacesub5 spine

After §0, the live recipe moved to **noweapon + surfacesub5** (isotropic
AVOID 55×55, surface-sub dodge 5×5, φ8, no blocking/weapon extras). Raw
syncs (shaped / skillmech provenance):
`tmp/ego_min_surfacesub5_clipsweep/`, `tmp/ego_min_surfacesub5_distabl/`,
`tmp/ego_min_surfacesub5_distbands/`, `tmp/ego_min_surfacesub5_cliprand/`,
`tmp/ego_min_surfacesub5_skillmech_rw/`.

### Live lock (sparse C/D/A, 2026-09-16) — **noclip**

**Locked:** `EGO_DISTANCE_CLIP_MAX_PX: null` (**noclip**) on noweapon +
surfacesub5 + φ8 + **sparse C/D/A** (`COLLECT_SPARSE_ONLY` /
`AVOID_SPARSE_ONLY`, `DESTROY_LIVING_PENALTY: false`). Stem:
`seaquest_skills_ego_pov_noclip_facing_psibins8_v2_noblocking_noweapon_surfacesub5_skillmech_sparse_cda`.
Sync: `tmp/cluster1_sparse_cda_clip_extremes/`. Details:
`docs/DISTANCE_CLIP_SAUCE.md`.

Rationale: under sparse rewards clip10/50/100 were already close on SQ home;
extremes showed **clip0 dead** and **noclip best home (~27k)** with healthy
mods vs PQN. Accepted ZS tax: Freeway (original clip motivation), softer
NTG/Breakout. No more clip sweeps on this spine.

Clip10 remains the near-field **control** cite; clip50/100 are ablations only.

### Provenance (pre-sparse skillmech) — clip10

**Historical lock:** **clip10** + ungated skillmech rewards on the same ego
spine (SQ ~27k in `tmp/ego_min_surfacesub5_skillmech_rw/`). Near-field
information-bound story for shaped COLLECT transfer. Do not reopen that
sweep; superseded for the sparse clean story by noclip above.

**Attribution baseline (same scaffolding):** suite
`fixed_ego_skill_baseline` keeps heuristic / labels / facing-frame obs /
POV mapper identical and replaces only skill Q-heads with
`fixed_ego_rules` (φ-servo) or `constant_ego` (FORWARD/FIRE/BACKWARD).
Evals are **base only** (`PAPER_EVAL_MOD_KEYS: [base]`). Materialize + run:

```bash
python scripts/materialize_fixed_ego_skill_baseline.py
python scripts/run_paper_eval.py --suite fixed_ego_skill_baseline --gpus 0 --lean
```

**Stop:** no more clip sweeps, bins, hard bands, train-time clip DR, or
distance-pose / can_fire encoding spam on this spine. Next is heuristic
smoke across ALE envs (§5.2–3) then the matrix (§3).

Lean (historical): clip10 vs clip50 meeting table kept below for provenance.
Skillmech refresh of the then-locked clip10 cell:

| cell | pre-ungate clip10 | **skillmech clip10** |
|------|------------------:|---------------------:|
| SQ base | 18 640 | **27 253** |
| Pong ZS | 0.0 `[0×3]` | 5.0 `[+15, 0, 0]` |
| BR ZS | 555 | **1 302** |
| PH ZS | 1 890 | **3 920** |

### Transfer + in-domain (clipsweep)

| clip | SQ base | Pong ZS | BR ZS | PH ZS | Pong episodes | Pong pts (P/E) |
|------|--------:|--------:|------:|------:|---------------|----------------|
| **clip10** | 18 640 | **0.0** | 555 | 1 890 | `[0, 0, 0]` | **20.3 / 20.3** |
| clip20 | 27 310 | −15.0 | 1 660 | **5 970** | `[−15×3]` | — |
| clip30 | 22 037 | −0.3 | 525 | 427 | `[+21, −11, −11]` | fluke mean |
| **clip50** | **35 333** | −3.7 | **1 939** | 5 093 | `[−20, +20, −11]` | 10.7 / 14.3 |
| clip80 | 24 860 | −0.3 | 940 | 4 750 | `[+21, −11, −11]` | fluke mean |
| clip100 | 27 473 | −11.0 | 1 099 | 3 390 | `[−11×3]` | — |
| noclip | 26 190 | −13.7 | 485 | 2 023 | `[−13/−14]` | — |

### Robustness — own-base category median retention

Retention = `mod_mean / agent_base`. Visual is always 1.00 on
`random_color_enemies` (geometry-only ego). **Caveat (MODS_PLAN §4.6):**
when stress mods land in a similar *absolute* band across clips while bases
differ (18k↔35k), own-base dynamics retention ranks ≈ `1/base`. Do **not**
read clip50’s 0.46 as “less robust than clip10’s 0.97” without the absolute
table below. Prefer abs stress cells (and vs PQN) for the thesis robustness
story once the clip is locked.

| clip | SQ base | dynamics med. | semantic med. | visual |
|------|--------:|--------------:|--------------:|-------:|
| **clip10** | 18 640 | **0.97** | 1.00 | 1.00 |
| clip20 | 27 310 | 0.78 | 1.13 | 1.00 |
| clip30 | 22 037 | 0.79 | 1.02 | 1.00 |
| **clip50** | 35 333 | 0.46 | 1.00 | 1.00 |
| clip80 | 24 860 | 0.56 | 0.91 | 1.00 |
| clip100 | 27 473 | 0.60 | 0.48 | 1.00 |
| noclip | 26 190 | 0.61 | 0.91 | 1.00 |

### Stress mods — absolute returns (same cells)

| mod | clip10 | clip50 | note |
|-----|-------:|-------:|------|
| gravity | **19 413** | 17 340 | clip10 slightly higher abs |
| faster_enemies | 690 | **1 670** | clip50 better |
| fast_oxygen_drain | 8 500 | **15 867** | clip50 better |
| shift_lanes | **20 123** | 19 757 | ≈ tied |
| surface_sub_always | 11 013 | **16 550** | clip50 better |
| random_color_enemies | 18 640 (1.00) | 35 333 (1.00) | appearance OK both |

### Encoding / DR probes on the same spine (rejected as lock)

None beats the clip10↔clip50 tradeoff. Keep as negative evidence only.

| arm | SQ | Pong | BR | PH | takeaway |
|-----|---:|-----:|---:|---:|----------|
| clip0 (dist zeroed) | 5 250 | −18.3 | 293 | 240 | collapses; direction-only fails |
| distbins8 | 14 447 | −19.0 | 772 | 1 700 | middling; no unlock |
| distbands3 (near/mid/far) | 17 180 | −13.7 | **2 771** | 2 293 | best BR; Pong still lost |
| clip50_sticky (train p=0.25) | 15 360 | −1.3 | 1 341 | 4 740 | Pong near-ties; SQ halves |
| cliprand `{10…80}`/ep | 3 960 | −13.7 | 293 | 4 947 | **clear negative** (SQ eps `[12k,0,0]`) |

### Meeting read

| Criterion | Prefer clip10 | Prefer clip50 |
|-----------|---------------|---------------|
| SQ competence | ~18.6k (PQN-parity band) | **35k ceiling** |
| Pong / COLLECT story | **consistent 20–20 ties** | noisy; often wrong-way paddle |
| Homologous DESTROY (BR/PH) | weak BR, OK PH | **best BR + strong PH** |
| Robustness *narrative* | dynamics retention ≈1; abs already competitive on gravity/lanes | better abs on O₂ / surface-sub / faster enemies; retention looks bad via base inflation |
| Thesis fit | tighter information bound | more in-domain capacity |

**Stop after the lock:** no more clip sweeps, bins, hard bands, or train-time
clip DR. Next is ALE heuristic smoke (`ale_clip10_heu_smoke`) then the
matrix (§3) on clip10.

---

## 1. Environment roles

### Bases (skill train from scratch + heuristic)

| Env | Why |
|-----|-----|
| **Seaquest** | Main robustness env; **all** skill input modes (ego / pixel / oc). |
| **Phoenix** | Second shooter layout; labels already exist; Space Invaders is a free homologous target. |
| **Name This Game** | Homologous underwater world (COLLECT + DESTROY). Policy: when the catcher/treasure field appears, **stick COLLECT** and keep it selected while it is active; dense reward for staying inside. Easy heuristic. |
| **Asteroids** | Rotate/thrust family base: continuous ship facing + `pov_rotate_aim` motor (LEFT/RIGHT turn). SQ→AS ZS allowed to fail (strafe≠turn). |

Time Pilot stays **transfer** (8-way homologous target for later AS→TP).
Do not promote both; one turn-family base is enough.

### Transfer targets (same list from every base)

| Env | Role |
|-----|------|
| Space Invaders | Phoenix-family DESTROY |
| Beamrider | already a win; keep |
| Time Pilot | omni DESTROY, 8-way; homologous turn-family target from Asteroids |
| Pong + Breakout | intercept family (drop Tennis) |
| Freeway | limitation (single-target vs gap) |
| Name This Game / Phoenix / Seaquest / Asteroids | cross-base (when not the source) |

**Deferred:** Kaboom (cleaner 1D catch / lateral-null) — not in upstream
jaxatari yet; label/policy code is parked unregistered.

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

- **Time Pilot / Asteroids:** POV facing = **real ship nose** (facing-frame
  bins: 16-way AS, 8-way TP). Shared 18-dim ego layout with Seaquest, but the
  POV mapper is **rotate-to-aim** (`pov_rotate_aim`): LEFT/RIGHT turn, FORWARD
  thrusts (AS) or holds heading (TP). Transfer and from-scratch use the **same**
  mapper + facing-frame flags. SQ→AS/TP ZS is still allowed to fail (strafe ≠
  turn); AS↔TP homologous transfer can succeed.

**Story we want:** SQ → TP/Asteroids ZS is allowed to fail (strafe ≠ turn).
Scratch-train one of them as a **base** with the rotate-aim motor, then transfer
to the other. If that works, skills learned rotate-and-thrust; they just were
not born with it on Seaquest. Homologous transfer inside a family, not a
failed Seaquest transfer.

---

## 3. Run matrix

Budget: **hard `19_988_480`** agent steps for every skill / PQN / tandem
train in the live matrix (batch-aligned “20M”; never mix with legacy
`24_985_600` / “25M” cells). **5M** finetune later. Match PQN step count
to skills. Do not mix budgets inside one comparison column.

Cross-transfer ZS is **base env only** (no full mod grid per cell). A few
hours of eval, not a few days of train.

| Block | Where | Modes | Notes |
|-------|--------|-------|--------|
| Skill train | Seaquest | ego, pixel, oc | thesis: ego is better **here** |
| Skill train | Phoenix, NTG, Asteroids | **ego only** | universal finding, not a second pixel/oc bake-off |
| PQN baselines | all **bases** | pixel (+ oc if cheap) | required |
| PQN baselines | **transfer** envs | pixel, once | ceiling; one-time at the chosen step count |
| Scratch skills | all transfer envs | **ego** | pain point; 20M |
| Finetune 5M | every base → every transfer | ego | necessary ceiling vs ZS (later) |
| Finetune 5M | Seaquest → transfers | pixel / oc as well | “full modes only on SQ” (later) |
| RQ3 tandem | all smoke envs this tranche | pixel meta + ego skills **from scratch** | warmstart skills→tandem deferred; **no meta-only** |
| RQ3 eval | that env’s **full mod suite** | tandem | compositionality + robustness |
| RQ3 transfer | skills from tandem ckpt only | ego ZS + target heuristic | meta stays home; same grid as skills-only, to show tandem did not wreck System 1 |

**Minimal-ego ablation** (block 0) stays **outside** this matrix until the
spine is locked; do not spawn 3× pixel/oc copies of every strip.

---

## 4. Mods

**Superseded by `docs/MODS_PLAN.md` (2026-09-08).** That doc locks seven parent
categories derived from the ego pipeline's own assumptions, a 17-mod Seaquest
grid, 4–6 mod strips elsewhere, and the metric fixes the current table needs
(score caps, oracle normalisation, solvability gate). The sketch below is kept
for provenance.

### Original sketch

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

Phoenix/NTG/Pong/SI get the **same intent grid**, implemented with
whatever jaxatari already has, then a short implement-list. Do not clone
Seaquest mods that do not exist (`shift_lanes` on Pong is already
`shift_player`). Full catalogue = later brainstorm; this table is the
template. (Kaboom deferred until the env lands upstream.)

---

## 5. Order of work

1. **Minimal ego** trains + SQ/Pong/BR/PH eval → lock spine channels (§0). ✅
1b. **Distance clip lock** — sparse C/D/A → **noclip** (§0.1). ✅
2. **Labels + heuristics:** NTG / TP / Asteroids wired. ✅
3. **Next:** suite **`noclip_sparse_smoke`** / `_gpu0` / `_gpu1` / `_gpu2` —
   per env (bases + transfers): ego scratch 20M + flat PQN 20M +
   from-scratch pixel tandem 20M + home mod evals. Asteroids = 4th base;
   TP = transfer; Montezuma deferred; no FT / no warmstart tandem /
   no cross-ZS in this suite. AS/TP use facing-frame spines (θ_h purged).
   Materialize:
   `python scripts/materialize_noclip_sparse_smoke.py`
4. **Mods workshop** (categories above × env). Implement missing jaxatari
   mods only for the locked ~8–12.
5. **Cross-ZS grid** (every base → every target, ego, base mod only).
6. **FT 5M** on transfers (ego; SQ also pixel/oc FT).
7. **RQ3 warmstart** (skills→tandem) on envs that already show tandem
   competence; ZS the **skills** (not the meta).
8. **Montezuma L1** nice-to-have only (policy broken).
9. Write with Freeway as limitation, Pong/Breakout as intercept
    family, homologous DESTROY as the transfer win.
    (Kaboom joins the intercept family once the env is upstream.)

## Explicit non-goals

- More AABB / bin / LCR pose arms, and **no further clip sweeps** after §0.1
  lock (clip0 / distbins / distbands / cliprand already negative).
- Multi-object attention / dual-slot revival.
- Kangaroo as flagship (timing). Keep as leftover if L1 ever clears.
- RQ3 meta-only and transferring a learned meta.
- Full pixel/oc bake-off on every ALE title.
