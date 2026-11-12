# Transfer ablation board — how to improve scores

Synced from `cluster1:~/ConceptBasedLearning/outputs/models/paper_eval/eval/`
on **2026-09-05**. CSVs live in `tmp/cluster_transfer_ablations/`.

Focus: **honest facing-mapper zero-shot**, especially Seaquest→Pong (the
persistent failure mode). Broader RQ2 (Beamrider / Phoenix) already transfers
on the collectsurface spine — see thesis ABC conclusions.

---

## TL;DR

| Lever tried | SeaQuest | Pong ZS | Verdict |
|-------------|----------|---------|---------|
| **φ8 clip50 v2** (reference) | 22 600 | **+1** (21–20) | Best Pong transfer so far |
| **navigatesurface** (surface→NAVIGATE) | **29 323** | **+0.33** (≈21–20) | Keeps Pong; boosts SQ |
| collectsurface (surface→COLLECT) | 19 193 | **−21** | Poisons Pong COLLECT |
| sparse COLLECT (± DESTROY sparse) | 21.7k / 32.7k | −18 / −13.7 | No Pong unlock |
| restrained LCR×near/far + legal mask | **2 643** | **−21** | Source collapse |
| clip0 / distflags5 distance ablations | 7.3k / 14.0k | −21 / −19 | No Pong unlock |
| angbins + legal-mask *train* | 27 817 | −21 | Mask≠fix without facing spine |
| Pong from-scratch (collectsurface ego) | — | +0.67 | Native ceiling still weak |

**Bottom line:** stop reward-sparsity and coarse-pose arms. Prefer **v2 or
navigatesurface** as the transfer spine. Next gain is almost certainly
**action legality during training** and/or **obs–reward geometry**, not another
distance/angle binning sweep.

---

## 1. What the recent arms actually changed

### A. Restrained ego (`restrained_ego`)

- Pose → 5 one-hots (L/C/R × near/far); `EGO_LEGAL_ACTION_MASK=true` at train+eval.
- Gate was SQ ≥20k and Pong > v2.
- Result: SQ **2.6k**, Pong **0–21**. Encoding too lossy for DESTROY/COLLECT under
  facing control. **Dead end.**

### B. Collect sparse Arms 1–2 (already logged 2026-09-03)

- Same φ8 v2 spine; strip dense COLLECT approach (Arm2 also strips DESTROY urgency).
- Result: SQ holds or rises; Pong gets *worse* than v2. Dense approach shaping is
  not the sole Pong villain — removing it does not create lateral intercept.

### C. Surface label A/B (`navigate_surface_tandem_pixel`)

- Identical φ8 clip50 facing to collectsurface except:
  - collectsurface: `SEAQUEST_SURFACE_LABEL: collect`
  - navigatesurface: `SEAQUEST_SURFACE_LABEL: navigate`
  - v2: default surface labeling (not forced COLLECT)
- **collectsurface kills Pong (−21)** while **v2 (+1)** and **navigatesurface
  (+0.33)** stay competitive. Forcing the surface waypoint through COLLECT
  appears to teach a COLLECT policy that does not intercept under Pong’s
  fixed-facing / FORWARD→NOOP geometry.

### D. Distance encode ablations (clip0, distflags5)

- On collectsurface spine: neither restores Pong; clip0 also hurts Beamrider
  (601 vs 1336). Distflags5 Beamrider 1543 is fine but Pong still −19.

### E. Legal mask on *angbins* (non-facing)

- SQ strong (27.8k) but Pong still −21. Confirms: **eval/train mask without the
  facing+φ8 spine does not fix intercept.** We still have **not** trained
  legal-mask on the φ8 facing v2 / navigatesurface checkpoint.

---

## 2. Diagnosis that still stands (Pong)

From `scripts/diagnose_pong_transfer.py` on v2
(`tmp/pong_transfer_diag/v2_phi8.json`):

- ~85% forwardish ego actions; ~42% map to **NOOP** on Pong (FORWARD into field).
- ~75% idle while `|lat|≥8` (should be strafing).
- Mean `|lat|` near paddle ≈ **16px** — starts late / stops short.
- Eval-only legal mask flips many argmaxes but **does not** close the miss.

So the policy’s COLLECT Q still prefers “close in depth / fire-ish” over
“null the lateral error,” and Pong makes depth-close a no-op.

---

## 3. Where transfer already works (don’t regress)

On collectsurface facing spine (thesis ABC):

| Env | ego ZS | Note |
|-----|--------|------|
| Beamrider | 1336 | ≈ scratch / near FT |
| Phoenix | 3453 | Large fraction of scratch |
| Freeway | 14.3 | Beats broken native FT/scratch |
| Tennis | −24 | Floor |
| Asterix | 0 | Needs FT/scratch |
| Pong | −21 | Floor on *this* spine; v2/navsurf recover |

Improving “transfer scores” for the paper claim should **preserve Beamrider /
Phoenix** while lifting Pong (and ideally Tennis). Do not swap to an encoding
that collapses SeaQuest (restrained) or that only helps Pong via env-specific
hacks.

---

## 4. Recommended next experiments (priority order)

### P0 — Legal-mask **train** on a competent facing spine

**Why:** diagnosis shows FORWARD Q buried under illegal (on Pong) actions;
restrained collapsed before we could test mask on a working policy; angbins+mask
lacked facing φ8.

**What:** clone `seaquest_skills_ego_pov_clip50_facing_psibins8_v2` *or*
`…_navigatesurface` with `EGO_LEGAL_ACTION_MASK: true` at **train** time
(Seaquest mapper still marks many actions legal — mask mainly reshapes Q for
transfer envs). Eval Pong/Tennis/Beamrider/Phoenix.

**Success:** Pong mean ≫ −21 and ≥ v2 (+1); Beamrider/Phoenix not worse than
collectsurface ZS by much.

### P0 — Prefer navigatesurface (or v2) over collectsurface for transfer spine

**Why:** surface→COLLECT is an avoidable own-goal on Pong; navigatesurface
matches v2 Pong and raises SQ base.

**What:** re-point thesis transfer / FT stems at navigatesurface (or keep v2)
before more Pong probes. Re-check Beamrider/Phoenix ZS on navigatesurface
(currently only Pong was eval’d in that suite).

### P1 — Obs–reward geometry consistency

From `docs/reward_shaping_BACKLOG.md`: dense COLLECT/NAVIGATE use
**centre–centre** distance; ego obs uses **AABB surface gap**. Retrain with
shared surface-gap shaping (not Pong-specific).

**Why:** late-stop / short intercept matches “reward says closing while hull
gap stalls.”

### P1 — Targeted COLLECT lateral pressure (still env-agnostic)

If mask-train is insufficient: small penalty on `|lat|` (body frame) when near,
or bonus for reducing `|lat|` — same signal Seaquest COLLECT needs for lane
align and Pong needs for intercept. Avoid Pong-only rewards.

### P2 — Do **not** schedule next

- More COLLECT/DESTROY sparsity arms
- Coarser pose (LCR / 4-way φ) until a mask-train continuous baseline works
- More clip/distflags sweeps aimed at Pong
- Relying on eval-only legal mask as the paper fix

### P2 — Parallel diagnostic (cheap)

Re-run `diagnose_pong_transfer.py` on **navigatesurface** vs **collectsurface**
vs **v2** (mask on/off). Expect collectsurface to show even worse idle/lat
stats; navigatesurface ≈ v2.

---

## 5. Numbers appendix (base means)

### Seaquest full mod eval

| Stem | base | median retention\* |
|------|------|--------------------|
| v2 φ8 clip50 | 22600 | 0.936 |
| navigatesurface | 29323 | 0.728 |
| collectsurface | 19193 | 1.000 |
| sparse1 | 21657 | 0.996 |
| sparse2 | 32727 | 0.857 |
| distflags5 | 14020 | 0.977 |
| clip0 | 7293 | 1.050 |
| lcr_nearfar | 2643 | 0.953 |

\*median over non-base mods; farming mods can inflate — use with care.

### Pong zero-shot (player–enemy when available)

| Stem | mean | scoreline |
|------|------|-----------|
| v2 | +1.00 | 21–20 |
| navigatesurface | +0.33 | ~21–20 |
| scratch collectsurface | +0.67 | — |
| sparse2 | −13.7 | ~7–21 |
| sparse1 | −18.0 | ~3–21 |
| distflags5 | −19.0 | — |
| collectsurface / clip0 / lcr / angbins+mask | −21.0 | 0–21 |

---

## 6. Checkpoints

```
# Best Pong reference
outputs/models/paper_eval/rq1/seaquest_skills_ego_pov_clip50_facing_psibins8_v2/...

# Strong SQ + nearly-v2 Pong
.../seaquest_skills_ego_pov_clip50_facing_psibins8_navigatesurface/...

# Thesis spine (good BR/PH, bad Pong)
.../seaquest_skills_ego_pov_clip50_facing_psibins8_collectsurface/...
```

Cluster path prefix: `cluster1:~/ConceptBasedLearning/`.
