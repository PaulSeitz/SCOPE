# Skill input modes (System 1)

This document describes how skill Q-networks are conditioned on *which object*
they act on, and why the cues are **index-invariant**.

`target_idx` is always a **gather key** (select bbox / mark a row / map ego
actions). It is **never** fed to the network as a bare slot one-hot. That matters
for transfer: RAM / layout slot `i` can mean different entities across games.

## Modes at a glance

| `SKILL_INPUT_MODE` | Observation | Target cue | Slot one-hot? |
|---|---|---|---|
| `ego` | Relative engineered features for the selected object | Built into the features (distance, angle, …) | No |
| `ego_embed` | Same as `ego`, then `Dense → ReLU` | Same as `ego` | No |
| `oc_embed` | Full labeled `objects` matrix (same leaves as meta) | Shared per-object `Dense`, then **gather target row** | No |
| `pixel_embed` | CNN over pixel stack | **Red outline** (bright ring if grayscale) on the target bbox | No |
| `pixel_embed_cond` | Alias of `pixel_embed` | Same outline (legacy name; slot-channel cond removed) | No |

## Ego (`ego` / `ego_embed`)

```
target_idx ──gather──► relative features (dx, dy, …) ──► MLP ──► Q(ego actions)
```

- Index only selects which object's geometry enters the feature builder.
- The Q-net never sees the integer slot id.
- This is the representation the paper argues for as transferable System-1 input.
- `ego_embed` adds an optional Dense projection (`SKILL_EMBED_DIM`).

## Object-centric (`oc_embed`)

```
objects ──► [M, F]   (all labeled leaves; F shared across envs)
              │
              ▼
     Dense(SKILL_EMBED_DIM) per row   ← shared weights, input always F
              │
              ▼
         [M, D] ──gather(target_idx)──► [D] ──► concat sin(ψ), cos(ψ) ──► MLP ──► Q
```

- When `SKILL_OC_PSI_FEATURES: true` (POV facing-relative mapper ablation), ψ is
  player facing minus bearing to the gathered target — same convention as POV ego
  `sin/cos(ψ)`. This tells the head whether the player is looking toward or away
  from the target under body-relative control, without switching to target-aligned
  FORWARD semantics.

- Uses the **entire** labeled object observation (bbox, active, visual_id, state,
  orientation, semantic_label, flags, …) — the same object leaves meta attaches
  per candidate. Skill does **not** also concat a global raw RAM flatten.
- **Per-object Dense, then gather** (not `Dense(M·F → D)` on a flat vector):
  - Dense kernel is `(F → D)`, so it loads across games when the object *leaf
    schema* is shared, even if `M` / `MAX_OBJECTS` differs (rows are padded).
  - Gathering the target row replaces slot one-hots / flattening `is_target`
    into a giant vector — “which object” is selection, not a portable ID code.
- Relabel-all-objects uses the same shared Dense and gathers `K` target rows.

Implementation: `build_oc_object_matrix` + `gather_object_rows` + optional
`build_oc_psi_features` in `observations/target_conditioning.py`; Dense lives in
`QNetwork`.

### Meta `META_INPUT_MODE: oc_embed`

Meta concatenates the full object matrix onto a global raw flatten. Meta
`oc_embed` does **not** use a Dense on that global stream (same as `flat` for
the global part). The transferable per-object embed lives on **skill**
`oc_embed` only.

### Meta `META_INPUT_MODE: oc_simple*`

Slim meta encodings for robustness ablations (`per_object_mlp` only). Suite:
`seaquest_meta_oc_ablation`.

| Mode | Global | Per slot |
|---|---|---|
| `oc_simple` | player xywh + O₂ + collected_divers | absolute xywh + active + label OH |
| `oc_simple_rel` | same | Δx, Δy, dist, same_lane, lane_delta, active + label OH |
| `oc_simple_role` | simple + role counts/min-dist | same as `oc_simple` |
| `oc_simple_rel_role` | simple + role pool | same as `oc_simple_rel` |
| `oc_simple_inv` | simple + lives + active COLLECT count | same as `oc_simple` |

**Omitted across the family:** `visual_id`, orientation/state, raw RAM flatten,
score digits. NOOP / FIRE reserved slots and the Seaquest surface waypoint stay
in the object matrix; masking still controls selectability.

## Pixels (`pixel_embed`)

```
pixel_obs + objects[target_idx].bbox
        │
        ▼
 paint outline (scaled into the pixel frame, incl. 84×84 resizes)
        │
        ▼
      CNN ──► Dense ──► MLP ──► Q
```

- Outline is drawn in **observation pixel space** (bboxes scaled from native
  160×210 Atari coords). On RGB frames the outline is red; on grayscale it is a
  bright ring.
- Relabel-all-objects broadcasts the frame per candidate target and paints each
  outline independently.
- Set `SKILL_PIXEL_PAINT_TARGET_OUTLINE: false` to feed the CNN raw normalized
  pixels with **no** target ring (ablation: transfer should collapse if the
  ring was the portable cue). Same Seaquest pixel checkpoint still loads.

Implementation: `paint_target_outline_pixels` / `normalize_pixel_obs` in
`observations/target_conditioning.py`.

## What we removed (and why)

Previously `oc_embed` / `pixel_embed` concatenated `one_hot(target_idx, MAX_OBJECTS)`
onto a global embedding (and `pixel_embed_cond` tiled a learned slot embed into
the CNN). That made ablations **fail under slot remapping by construction**.

A flat `Dense(M·F → D)` was also insufficient for transfer: the first-layer
input width still depended on `M`. The per-object Dense + gather fixes that
when `F` is shared.

## Config knobs

See `configs/template_config.yaml`:

- `SKILL_INPUT_MODE`
- `SKILL_OC_PSI_FEATURES` — concat `sin(ψ), cos(ψ)` on skill `oc_embed` (POV facing mapper)
- `SKILL_EMBED_DIM` — used by `ego_embed`, skill `oc_embed` (per-object), pixel CNN proj
- `SKILL_PIXEL_CNN_CHANNELS` / `SKILL_PIXEL_CNN_KERNEL_SIZE`
- `SKILL_PIXEL_PAINT_TARGET_OUTLINE` — paint target ring for `pixel_embed` (default true)
- `SKILL_PIXEL_TARGET_EMBED_DIM` — unused (kept for config compatibility)
- `META_EMBED_DIM` — used by meta `pixel_embed` (not meta `oc_embed`)

Pixel skill modes still require `PixelAndObjectObsWrapper` (same as before).
