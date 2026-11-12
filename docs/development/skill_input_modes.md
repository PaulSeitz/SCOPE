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
| `ego_scene_rel` | `ego` + every slot in the same relative encoding | `ego` prefix + `is_target` column | No (mark, not one-hot input id) |
| `ego_abs_pos` | `ego` + player centre (x/160, y/210) per frame | Same as `ego` | No |
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

## Further ablations (`ego_oc_native` / flat `ego_scene_rel_flat`)

- `ego_oc_native` (skill arm `skills_ego_ocnative`, needs `OC_NATIVE_STACK: true`):
  `ego(target) || native vector || one_hot(target)`. Tests whether absolute object
  coordinates and the extra native fields still harm layout robustness when the
  target-relative input is present.
- `ego_scene_rows` (skill arm `skills_ego_scenerows`): the `ego_scene_rel` rows without the
  reference prefix. The target is identified by `is_target` only.
- `OC_INPUT_MODE: ego_scene_rel_flat` (flat PQN arm `pqn_ego_scenerel`): the scene rows
  without `is_target` and without the prefix, the 18-action facing-relative mapping and
  a 3x128 MLP, trained on the environment reward. Isolates the hierarchy against
  `skills_ego_scenerel`.

## Context ablations (`ego_scene_rel` / `ego_abs_pos`)

Thesis follow-up to separate *what* the restriction withholds. Both keep the
reference 36-value `ego` vector as an unchanged prefix, and the same selector,
rewards, ego action mapping, 3×128 MLP and optimisation. Source:
`observations/ego_context.py`; configs: `scripts/materialize_context_ablation.py`
→ `configs/experiment/paper_eval/context_ablation/`.

```
ego_scene_rel:  ego(target) || for j in slots: [ego(j)_f·active_f, active_f]×4 , is_target_j
ego_abs_pos:    ego(target) || [px_f/160, py_f/210]×4
```

- `ego_scene_rel` uses the reference per-object encoding (AABB-gap distance +
  facing-relative ψ bins) for every slot, so the only change is *which objects*
  the skill sees. No absolute coordinates, player box, labels or oxygen
  (unlike `oc_flat_rel`, whose global tail carries the absolute player box).
  Width: `36 + M·41` (Seaquest M=35 → 1471, Name This Game M=11 → 487). It
  depends on M, so this arm does not transfer across games.
- Caveat: fixed or lane-bound slots make absolute position recoverable from
  relative offsets. Seaquest's surface waypoint (slot 33, always active, y=44)
  gives depth, and the Name This Game tentacles (slots 1–8, x std ≈ 4.5 px)
  give x. The arm removes *explicit* absolute location, not its correlates.
- `ego_abs_pos` appends the explicit player centre for each stacked frame
  (width 44). If the policy ignores it, robustness can stay unchanged. That
  result would weaken the claim that withholding absolute location is necessary.
- Tests: `tests/test_ego_context.py` (prefix identity, masking, rigid-shift
  invariance of the scene, exact abs-pos tail) on real Seaquest/NTG states.

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
