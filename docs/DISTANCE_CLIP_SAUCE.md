# Distance clip / ego ablation — keep the sauce

Date: 2026-09-11. Working notes so we don’t re-litigate this cold.

## Spine (frozen)

- Facing-frame POV mapper, **φ8** (unless a named sincos probe), surfacesub5
  labeling (isotropic AVOID 55×55, surface-sub dodge 5×5).
- **Off:** size, velocity, blocking, `facing_enemy`, shot geometry.
- **Open / re-opening:** distance encoding + whether **`can_fire` only** is on
  (`USE_EGO_WEAPON_FEATURES: true`, `USE_EGO_FACING_ENEMY_FEATURES: false`).

Blocking and facing_enemy are redundant — drop permanently.

## The clip problem (main issue)

Clip is a **saturation horizon** on a 1-D range channel, not a smooth capacity dial.

Sweep on noweapon+surfacesub5 did **not** give a clean
“higher clip → more SQ / worse transfer” curve. What it supports:

| Regime | ~clip | Behavior |
|--------|------:|----------|
| **Near-field** | ~10 | Distance saturates fast → φ + local range. Pong COLLECT stable (`[0,0,0]` / 20–20). SQ competent (~18k) but not farmy. BR/PH weak. |
| **Operational** | ≳20 | Mid-range distance usable for SQ farming → SQ/DESTROY up; Pong never stably transfers again (dead or lottery). Peak SQ ~35k at **clip50**. |

So clip10 and clip50 are **regime exemplars**, not two random lucky spikes.
Middle clips (20/30/80) are noisy samples inside the operational regime where
Pong is already broken — fluke means are not second transfer peaks.

**Thesis framing:** lock representation (scalar dist + φ8 + surfacesub5);
report clip as a sensitivity / regime parameter. Prefer **clip10** for the
information-bound / COLLECT transfer story; cite **clip50** as competence ceiling.

## Robustness trap

Own-base median retention ≈ `mod / base` ranks **worse agents higher** when
stress mods hard-cap in a shared absolute band while bases differ (18k↔35k).
Clip50 / neartiers dyn~0.46 vs clip10~0.97 is largely that artifact.
**Prefer absolute stress cells** (and vs PQN) for the robustness story
(`MODS_PLAN` §4.6).

## What we tried (encoding) — rejected as lock

| Arm | Takeaway |
|-----|----------|
| clip0 / distbins8 / distbands3 | No COLLECT unlock; bands help BR only |
| cliprand / sticky | DR / sticky ≠ free lunch |
| Δxy / sign / posxy | All lose Pong hard; no lock |
| **neartiers** (clip10 continuous + mid/far one-hots) | **First SQ+Pong stitch:** SQ~34k, Pong −1×3 (20–21). BR/PH still floor. Abs mods weaker than old can_fire arm. |

## `can_fire` history

EVAL_PLAN §0 already locked **noblocking + can_fire only** (SQ 34k, PH 4.6k)
before surfacesub5 drifted to **noweapon**. Absolute dynamics stress looked
better with can_fire than with high-base noweapon/neartiers. Re-intro on the
surfacesub5 spine is the overnight question — not a new combinatorial axis
of encodings.

## Overnight batch (2026-09-11) — done

Suite `ego_min_surfacesub5_canfire_overnight` (cluster1): can_fire re-intro
on clip10/50/neartiers/Δxy/Δxy+sincos was a **clear negative** on this spine
(Pong stories died; clip50 SQ collapsed). Do not fold can_fire back into the
surfacesub5 lock.

**Lock: clip10 + noweapon + surfacesub5 + φ8.** Next: ALE heuristic/skill
smoke (`ale_clip10_heu_smoke`), then matrix.
