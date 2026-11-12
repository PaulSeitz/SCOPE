# Distance clip / ego ablation — keep the sauce

Date: 2026-09-11 (sparse C/D/A lock amended 2026-09-16). Working notes so we
don’t re-litigate this cold.

## Spine (frozen)

- Facing-frame POV mapper, **φ8** (unless a named sincos probe), surfacesub5
  labeling (isotropic AVOID 55×55, surface-sub dodge 5×5).
- **Off:** size, velocity, blocking, `facing_enemy`, shot geometry.
- **Reward (live story):** sparse C/D/A —
  `COLLECT_SPARSE_ONLY` / `AVOID_SPARSE_ONLY`, `DESTROY_LIVING_PENALTY: false`
  (NAVIGATE stays shaped). Older shaped/ungated skillmech tables below are
  provenance.

Blocking and facing_enemy are redundant — drop permanently.

## Sparse C/D/A lock (2026-09-16) — **noclip**

Suite sync: `tmp/cluster1_sparse_cda_clean/`, `tmp/cluster1_sparse_cda_clip_transfer/`,
`tmp/cluster1_sparse_cda_clip_extremes/`.

Under sparse rewards the old clip10↔clip50↔clip100 home band was already
tight (~19–22k). Extremes closed the question:

| Arm | SQ base | Verdict |
|-----|--------:|---------|
| **clip0** | **0** | Dead (no kills/collects). Sparse needs *some* distance. |
| clip10 | 18 873 | Balanced ZS backup (Freeway/Breakout/NTG friendlier). |
| clip50 / clip100 | 20–22k | Ablations; clip100 helps SI/TP, kills Phoenix. |
| **noclip** (`EGO_DISTANCE_CLIP_MAX_PX: null`) | **27 291** | **Lock** — best home, healthy mods vs PQN, no magic clip number. |

**Ckpt:**
`outputs/models/paper_eval/rq1/seaquest_skills_ego_pov_noclip_facing_psibins8_v2_noblocking_noweapon_surfacesub5_skillmech_sparse_cda/seaquest/hierarchical_pqn_seaquest_seed44.safetensors`

**Transfer tax (accepted):** Freeway ≈0 (original motivation for clipping);
NTG soft (~880 vs clip10 ~1.9k — erratic long-range tentacles plausible);
Breakout soft. Shooters (SI/TP/AS/BR) ≥ clip10. Pong stays a hard-fail under
sparse COLLECT (policy, not a clip reopen).

**Mod quirks (not blockers):**
- `unlimited_oxygen` < base for noclip: surfacing cadence collapses
  (`surface_events` ~0.4 vs ~24 on base) → shorter episodes / fewer kills.
  Removing O₂ pressure breaks a loop the strong agent learned; not “O₂ should
  help.”
- `fast_oxygen_drain` is the main absolute robustness dip (~5.7k, heavy
  surface thrash). Ratio-to-base looks worse because base is high — prefer
  absolute cells + vs PQN (`MODS_PLAN` §4.6).
- `lethal_divers` → 0 on noclip (clip10 still ~3k): note in text, do not reopen clip.

**Stop:** no more clip sweeps on the sparse spine. Cite clip10 as the
near-field control; cite clip0 as the negative (distance is load-bearing).

**Next matrix (not a clip reopen):** suite `noclip_sparse_smoke` — hard
`19_988_480` steps everywhere (ego scratch / flat PQN / from-scratch
tandem). Do not mix with the 25M noclip lock ckpt above.

## The clip problem (shaped / pre-sparse provenance)

Clip is a **saturation horizon** on a 1-D range channel, not a smooth capacity dial.

Sweep on noweapon+surfacesub5 did **not** give a clean
“higher clip → more SQ / worse transfer” curve. What it supports:

| Regime | ~clip | Behavior |
|--------|------:|----------|
| **Near-field** | ~10 | Distance saturates fast → φ + local range. Pong COLLECT stable (`[0,0,0]` / 20–20). SQ competent (~18k) but not farmy. BR/PH weak. |
| **Operational** | ≳20 | Mid-range distance usable for SQ farming → SQ/DESTROY up; Pong never stably transfers again (dead or lottery). Peak SQ ~35k at **clip50**. |

So clip10 and clip50 were **regime exemplars** under shaped/skillmech rewards,
not two random lucky spikes. Under **sparse** C/D/A that regime split largely
collapsed — hence the noclip lock above.

**Historical thesis framing (pre-sparse):** prefer clip10 for COLLECT transfer;
cite clip50 as competence ceiling. **Live framing:** prefer **noclip** for the
sparse clean story; report Freeway as the known limitation of unclipped range.

## Robustness trap

Own-base median retention ≈ `mod / base` ranks **worse agents higher** when
stress mods hard-cap in a shared absolute band while bases differ (18k↔35k).
Clip50 / neartiers dyn~0.46 vs clip10~0.97 is largely that artifact.
**Prefer absolute stress cells** (and vs PQN) for the robustness story
(`MODS_PLAN` §4.6). Same trap applies to sparse noclip (27k) vs clip10 (19k).

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

**Pre-sparse lock (superseded for sparse story):** clip10 + noweapon +
surfacesub5 + φ8. **Live sparse lock:** **noclip** + noweapon + surfacesub5 +
φ8 + sparse C/D/A (see top).
