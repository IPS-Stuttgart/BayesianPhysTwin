# Deform360 `001-rope` public-data protocol

This track avoids waiting for gated PokeFlex access while leaving the frozen
Bayesian-PhysTwin and pre-acquisition protocols unchanged. It uses only the ten
public `001-rope` episodes.

## Pinned inputs

- Code: `https://github.com/lhy0807/deform360` at
  `0fe36f0b7a7a917ba62b5f8cee707299a9a4a317`.
- Dataset: `brownu/deform360` at
  `7fea8e20231a47641d1d2bc8791920ec4e62ec5e`.
- Object: `raw/001-rope` only.
- Audio `.wav` and `.flac` files are omitted because the released tactile NPY
  arrays and timestamp sidecars are the authoritative processing inputs.

The expected non-audio cohort has 908 files: 41 camera streams with ten exact
MP4/timestamp pairs, four tactile streams with ten exact NPY/timestamp pairs and
one median baseline each, three calibration dictionaries, and `metadata.json`.

## Frozen split

Episodes are ranked by SHA-256 of
`causal4d-deform360-001-rope-v1:<episode-id>`, without reading frames, tactile
values, tracks, geometry, or prediction errors.

| Split | Episode indices | Actions |
| --- | --- | --- |
| Source | 0, 2, 3, 4, 5, 7, 8 | move edge; lift edge; lift center; curl edge; lift both edges; push both edges; curl both edges |
| Calibration | 1, 9 | move center; lift middle |
| Target | 6 | move both edges |

The target action is therefore frozen before preprocessing or model fitting.

## Information boundary

Three comparisons use progressively stronger evidence:

1. `visual_only`: RGB/geometry and permitted prefix robot-pose evidence, with no
   tactile values.
2. `tactile_conditioned_z`: tactile evidence from source/calibration episodes
   and only the permitted target prefix. The six-frame prefix begins at a
   causal robot-opening trigger fitted on source/calibration episodes; target
   tactile is not used to choose its location. For the bimanual target action,
   all target grippers must satisfy the trigger for three consecutive frames.
3. `oracle_tactile`: the full target tactile contact event, opened only after
   both preceding predictions are sealed. It is an upper-bound diagnostic, not
   a deployable method.

The public tactile signal is unitless and peak-normalized, not calibrated force.
The robot trajectory is vision-recovered/measured rather than a separately
logged command. Tangential slip is not observed. Claims are restricted to
contact timing/assignment and held-out interventional prediction; controller
gain, command-to-realization delay, calibrated wrench, and slip ground truth are
out of scope.

For the action-conditioned future-prediction benchmark, the released future
robot trajectory may be supplied as conditioning evidence, matching the
Deform360/PhysTwin evaluation setting. It must not be described as a commanded
control stream or used to estimate command-to-realization error.

Camera-track evidence carries a frozen synchronization reliability weight
derived before prediction:

```text
(1 - reused-frame fraction) *
exp(-0.5 * (p95 timestamp error / 33333 us)^2)
```

This prevents a nominally synchronized but heavily frame-repeated view from
receiving the same pseudo-measurement weight as a native-rate view.

## Commands

Run the locked preflight before model fitting:

```bash
causal4d-deform360-preflight \
  /mnt/lexar4tb/datasets/deform360/data-7fea8e2/raw/001-rope \
  results/causal4d_public/deform360_001_rope_preflight.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json \
  --hash-media
```

After the official synchronization and tactile stages, pass the processed root
with `--processed-root`. `--unlock-target-prefix` exposes only the locked six
target frames to the tactile-conditioned method and requires the start frame
emitted by the sealed source/calibration robot-opening model, for example
`--target-prefix-start-frame 84`. Do not use
`--unlock-target-oracle` until visual-only and tactile-prefix target predictions
have been checksummed and sealed.

The contact comparison itself is staged so the oracle cannot be opened early:

```bash
causal4d-deform360-contact fit RAW_001_ROPE PROCESSED_ROOT contact_model.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json

causal4d-deform360-contact seal PROCESSED_ROOT contact_model.json \
  target_contact_predictions.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json
```

`fit` reads complete tactile streams only for source/calibration episodes.
`seal` chooses the target prefix from robot opening alone, reads exactly that
six-frame tactile slice, and emits no target metric. The final `evaluate`
subcommand requires the SHA-256 seal of the held-out future prediction before
it will read the full target tactile stream. The opening-only baseline is a
deliberately cheap visual/proprioceptive control; it is not presented as the
paper's transformer-based RGB contact predictor.

## Public SAM2 mask fallback

Deform360's released mask processor expects gated SAM3 weights. While those
weights are unavailable, source and calibration geometry may use a transparent
public fallback pinned to:

- SAM2 repository commit
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`;
- `sam2.1_hiera_small.pt` with SHA-256
  `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`;
- model configuration `configs/sam2.1/sam2.1_hiera_s.yaml`.

The adapter selects an automatic first-frame SAM2 mask that contains both
colored rope families and has a rope-like elongated support. A frozen
calibration check then retains only masks that contain the common 3D consensus
core reconstructed from the other views. This rejects plausible-looking robot
hardware masks without using a prediction metric. Accepted masks are propagated
through the video. The adapter uses the official Deform360 HDF5 mask writer,
but it is explicitly a public SAM2 fallback rather than a reproduction of the
gated SAM3 stage.

Run and seal the source-only view audit first:

```bash
causal4d-deform360-sam2-views PROCESSED_ROOT 0 source_episode_0_views.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json \
  --sam2-repository /path/to/pinned/sam2 \
  --checkpoint /path/to/sam2.1_hiera_small.pt
```

```bash
causal4d-deform360-sam2-masks PROCESSED_ROOT 0 source_episode_0_masks.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json \
  --sam2-repository /path/to/pinned/sam2 \
  --checkpoint /path/to/sam2.1_hiera_small.pt \
  --view-audit-json source_episode_0_views.json
```

Source and calibration episodes require no prediction seal. Full target masks
remain inaccessible through this command until
`--held-out-prediction-seal-sha256` names a valid sealed prediction. This is
independent of the six-frame target tactile prefix: opening tactile evidence
does not authorize future visual annotations.

Target state initialization uses a separate prefix-only command. Its candidate
camera set is frozen from a source view audit and source synchronization
reliability, and the frame bounds come from the checksummed contact-prediction
seal. It decodes exactly the six permitted target frames and writes standalone
NPY masks; it does not create a full-timeline `mask_refined.h5` that downstream
code could mistake for an unlocked target annotation. Every selected camera is
attempted, deterministic segmentation failures are recorded, and at least eight
successful cameras are required:

```bash
causal4d-deform360-sam2-prefix \
  PROCESSED_ROOT target_contact_predictions.json source_episode_0_views.json \
  deform360_001_rope_preflight.json target_prefix_masks \
  target_prefix_masks.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json \
  --sam2-repository /path/to/pinned/sam2 \
  --checkpoint /path/to/sam2.1_hiera_small.pt \
  --minimum-sync-reliability 0.85
```

Before paying for per-frame Splatfacto reconstruction, run the source-frame
thin-rope probe. The upstream default relaxes visual-hull consensus until it has
10,240 points, which can inflate a thin rope into a broad seed volume. This
probe keeps the pinned Splatfacto trainer but fixes a 256-point minimum and
audits robust PCA spans plus opacity-weighted projection containment:

```bash
causal4d-deform360-splat-probe \
  PROCESSED_ROOT source_episode_0_views.json \
  deform360_001_rope_preflight.json source0_splat_probe \
  source0_splat_probe.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json \
  --iterations 1000 --minimum-sync-reliability 0.85
```

The probe is source-only development QA. Passing it authorizes a reconstruction
run; it is not a target result and does not alter the frozen Deform360 code.

For the shared-dynamics pilot, full per-frame Splatfacto is not required. A
source-only centerline command first builds the tight coarse hull, then carves a
4 mm local multiview hull around the previous state and extracts a 21-node open
chain at normalized arc-length locations:

```bash
causal4d-deform360-rope-sequence \
  PROCESSED_ROOT 0 source_episode_0_views.json \
  deform360_001_rope_preflight.json source0_centerlines.npz \
  source0_centerlines.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json \
  --mask-audit-json source_episode_0_masks.json \
  --frame-stride 2 --minimum-sync-reliability 0.85
```

These nodes are silhouette-derived normalized-arc-length pseudo-correspondences,
not independently verified material identities. Ordered track error therefore
remains a secondary diagnostic; symmetric 3D Chamfer is the primary geometry
metric for this public-data pilot.

## Source observations and shared dynamics

Each source centerline sequence is paired with its source-only contact state,
vision-recovered controller trajectory, and a controller-to-rope contact offset:

```bash
causal4d-deform360-rope-observation \
  PROCESSED_ROOT contact_model.json source0_centerlines.json \
  source0_observation.npz source0_observation.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json
```

The frozen source quality gates accepted episodes `0,3,4,5,8`. Episode `2`
failed usable geometry under occlusion and episode `7` failed the available
point-contact representation for its nonprehensile action. Both exclusions
were made without target geometry, tactile, or errors.

Fit the finite 200-candidate forward grid and its leave-one-action-out check
using only the five accepted source observations:

```bash
causal4d-deform360-rope-fit \
  source0_observation.json source3_observation.json \
  source4_observation.json source5_observation.json \
  source8_observation.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json \
  --output shared_forward_fit.json
```

The model is an inextensible 21-node chain with kinematic contact constraints.
The selected reduced dynamics use bending acceleration `8.0`, contact damping
`5.0`, and drag `0.2` in the artifact's units; all other candidate coefficients
are zero. Effective gravity is fixed to zero as a support-balanced approximation
for this table-top reduced model, not as a claim that gravity is absent. Initial
velocity is fixed to zero because consecutive silhouette centerlines do not
provide verified material identities from which velocity could be estimated.

The source competence gate requires at least 5% pooled Chamfer improvement over
constant persistence and at least 60% leave-one-action-out wins. The selected
model passed with 25.17% pooled improvement and four wins in five episodes.

## Prefix geometry and prediction seal

The target prefix mask stage attempted the 21 source-locked cameras. Sixteen
passed; five deterministic mask failures were retained in the audit rather than
silently changing the camera policy. Reconstruct the six-frame target prefix
using only those masks and the accepted source centerline artifacts:

```bash
causal4d-deform360-rope-prefix \
  PROCESSED_ROOT target_prefix_masks.json \
  target_prefix_geometry.npz target_prefix_geometry.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json \
  --source-sequence-json source0_centerlines.json \
  --source-sequence-json source3_centerlines.json \
  --source-sequence-json source4_centerlines.json \
  --source-sequence-json source5_centerlines.json \
  --source-sequence-json source8_centerlines.json
```

The raw prefix hull under-recovered the heavily occluded rope at 264.5 mm. The
frozen correction preserves the observed centerline shape but resamples it to
the quality-passing, source-only same-object median length of 315.58 mm. This is
an explicit object-length prior; it does not use the target future.

Build and checksum both deployable open-loop rollouts before opening any target
future mask or full target tactile stream:

```bash
causal4d-deform360-rope-predict \
  PROCESSED_ROOT target_contact_predictions.json shared_forward_fit.json \
  target_prefix_geometry.json held_out_predictions.npz \
  held_out_predictions.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json
```

The immutable held-out prediction seal is
`add9b28154159f71e9bd7d631d68cbfd73e0b63ba8a487cedace5bead48ec667`.

## Post-seal oracle and evaluation

Only after the preceding seal may the full target tactile stream and target
suffix masks be opened:

```bash
causal4d-deform360-contact evaluate \
  PROCESSED_ROOT contact_model.json target_contact_predictions.json \
  target_contact_oracle.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json \
  --held-out-prediction-seal-sha256 \
  add9b28154159f71e9bd7d631d68cbfd73e0b63ba8a487cedace5bead48ec667

causal4d-deform360-rope-oracle \
  PROCESSED_ROOT held_out_predictions.json target_contact_oracle.json \
  shared_forward_fit.json target_prefix_geometry.json \
  oracle_tactile_prediction.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json

causal4d-deform360-sam2-suffix \
  PROCESSED_ROOT held_out_predictions.json target_prefix_masks.json \
  target_suffix_masks target_suffix_masks.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json \
  --sam2-repository /path/to/pinned/sam2 \
  --checkpoint /path/to/sam2.1_hiera_small.pt

causal4d-deform360-rope-future \
  PROCESSED_ROOT held_out_predictions.json target_prefix_geometry.json \
  target_suffix_masks.json target_future_geometry.npz \
  target_future_geometry.json \
  --config configs/causal4d_public/deform360_001_rope_v1.json

causal4d-deform360-rope-evaluate \
  held_out_predictions.json target_prefix_geometry.json \
  target_future_geometry.json oracle_tactile_prediction.json \
  held_out_evaluation.json
```

The target future contains frames `[109,237)` and is untouched by model fitting.
The field named `tactile_conditioned_z` in the post-seal contact-oracle artifact
is a full-stream contact diagnostic. It is not the deployable six-frame method;
only `oracle_tactile` from that artifact is used as the oracle rollout.

## Held-out result

| Method | Future CD (mm) | Future track (mm) | CD vs persistence |
| --- | ---: | ---: | ---: |
| Constant persistence | 71.84 | 78.87 | 0.00% |
| Visual-only contact | 47.58 | 60.16 | -33.77% |
| Six-frame tactile-conditioned `z` | 59.74 | 69.67 | -16.84% |
| Full-tactile oracle | 46.70 | 59.80 | -34.99% |

A source-pooled physical forward model transfers to the held-out
`move both edges` action and beats persistence. Constant persistence is only a
competence baseline: because this pilot did not run a matched single-source
fitting control, the target gain cannot yet be attributed specifically to
pooling. The fitted backend is the reduced inextensible centerline simulator
defined above, not PhysTwin/Warp or Bayesian-PhysTwin.

The six-frame tactile state is worse than visual-only because its second
gripper is inactive at prefix frame 108, while oracle contact begins at frame
109; the frozen open-loop policy cannot anticipate that transition. Visual-only
is within 1.84% CD and 0.60% track error of the full-tactile oracle.

This supports physical-forward-model competence for one held-out action and
rejects a static six-frame endpoint contact state for that action. It does not
establish a pooling benefit, population-level transfer, or that tactile is
generally unhelpful. Online tactile filtering and contact-transition inference
remain untested.
