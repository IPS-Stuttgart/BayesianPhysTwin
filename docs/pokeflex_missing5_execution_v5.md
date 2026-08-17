# PokeFlex Missing-Five V5 Execution Boundary

This execution layer completes the pretarget engineering for the five unavailable
official PokeFlex takes. It does not contain their outcomes and does not authorize
a result claim.

## Prediction inputs

The target take's upstream-selected template mesh is an explicit input to the
published reconstruction task. It is therefore allowed during prediction. Every
other mesh remains an outcome and is forbidden until the all-five prediction
barrier passes.

The `stage` command reads the exact author archive bound by the V4 source
manifest and writes a prediction-only tree containing:

- `robot_data.json`;
- both Kinect camera-parameter files;
- Kinect depth frames 1 through the frame before the final prediction;
- exactly one template mesh selected by the upstream rule.

The stage validator rejects an additional mesh even when the artifact is
re-signed. The prediction process receives only this staged tree, so future mesh
custody is structural rather than a statement about intended code behavior.

For target frame `f`, the checkpoint uses only Kinect depth frames `f-5` through
`f-1`. The update at `f-1` is transferred once to `f`. Unsupported updates are
byte-identical checkpoint fallbacks.

## Frozen arms

Each prediction archive contains four vertex trajectories over the same faces:

| Arm | Effective scale |
| --- | --- |
| Released checkpoint | 0 |
| Global source scale | 0.125 |
| Frozen V4 | Parent V4 object-specific scale |
| Frozen V5 | Source-selected missing-target scale |

The V5 scales are 0.125 for `Pillow_T8`, 0.25 for
`3dPrintedCylinder_T7`, 0.1875 for `3dPrintedHeart_T14`, 0.125 for
`Sponge_T10`, and 0.125 for `3dPrintedPizza_T13`.

No target score can change a scale, support decision, fallback, surface-sampling
seed, or gate.

## Artifact chain

1. The inherited V4 author-source manifest binds all five ZIP archives without
   decoding member payloads.
2. One typed input-stage artifact binds the allowed extracted members and proves
   that exactly one template mesh and zero future target meshes were decoded.
3. One compressed prediction archive and typed seal are produced per take.
4. The barrier validates all five seals, requires one clean implementation
   revision, and records zero target-mesh access and zero target metrics.
5. Only the `score` command accepts the complete barrier. It revalidates every
   seal and source archive before reading scored mesh members.
6. The result validator recomputes per-frame means, prospective V5-versus-V4
   gates, and the combined official-18 gates from immutable public-13 evidence.

The current-main integration lock was issued before any target archive became
available. Its canonical SHA-256 is
`1bc0a3486c0b937000772fc74bfcab2ed4a4dd34f2d90d0199440bc043a59f7a`
and its file SHA-256 is
`eaa1eea978783f25255fe3667a0222689c7055b128b4e2cfcf052aa9a7e27cf2`.

This lock supersedes the pre-integration execution digest
`62c8c9c81a5c3dd179864624ffca69769ace4a59c6e49c993aab42080b295064`.
The only bound implementation change is explicit static type narrowing after
the existing runtime validation checks; the numerical prediction, fallback,
surface sampling, scoring, and gate logic are unchanged. No author target
archive or target outcome was available or accessed during the amendment.

## Public parity control

The split implementation was replayed on the previously opened public take
`3dPrintedCylinder_T1`. The reusable prediction arrays were byte-identical to
the prior sealed runner for the checkpoint, global 0.125 arm, and 0.25 arm.
Faces, frame indices, update masks, action/robot support, and correction RMS were
also byte-identical.

Frame-by-frame scoring over all 97 active frames had exactly 0 mm numerical
difference for checkpoint, global, V4 0.375, and V5 0.25 arms. This is execution
parity evidence only; it is not target evidence. The compact record is in
`results/sota/pokeflex_missing5_execution_v5/public_parity_3dPrintedCylinder_T1.json`.
That compact record remains bound to the pre-integration execution digest.

The current-main lock was then replayed independently on the same public take at
implementation revision `f87d982c77778ffe6b8247ae01d7c0dd2a0a350b`.
All 13 reusable legacy prediction, topology, frame, support, and diagnostic arrays
were byte-identical. Frame-by-frame checkpoint, global, V4, and V5 scores again
differed by exactly 0 mm on all 97 action frames. The replay had 97 supported
prediction frames overall and 92 supported updates inside the scored action
subset. The compact current-lock record is in
`results/sota/pokeflex_missing5_execution_v5/current_main_public_parity_3dPrintedCylinder_T1.json`
with file SHA-256
`83f7a312d739e6974fcc861159c9797a56f7b22a7ae72fb75581a662a5e6ae46`.

## Claim boundary

The five author target archives are still unavailable. Their target meshes have
not been opened, no prediction seal exists for them, and no prospective or
official-18 gate has been evaluated. The exact next operation after archive
delivery is source-manifest validation followed by `stage` and `predict` for all
five takes. Scoring remains prohibited until the complete barrier validates.

No Deform360 held-v8 runtime, target, query, score, barrier, or outcome artifact
is part of this protocol.
