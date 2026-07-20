# Deform360 Query-Field Transfer V8

Run date: 2026-07-20

Status: frozen development decision; prospective held transfer in progress.
No official Deform360 state-of-the-art claim is made here.

## Research question

Can one shared, recursively corrected displacement field transfer a sparse
online belief to arbitrary official material identities more accurately than
the same sealed physical backbone without that correction?

The frozen method predicts a physical trajectory and updates it from 16 sparse
material identities at logical frames 19, 38, and 57. It then decodes both the
corrected trajectory and the uncorrected physical comparator at any frame-zero
3D query through the same Gaussian field:

- four nearest source nodes;
- Gaussian length scale `0.05` times the robust frame-zero object scale;
- support radius `0.50` times that scale;
- distance-then-source-ID tie breaking;
- bit-exact output at source anchors; and
- one shared support mask for both arms.

The field interpolates total displacement from frame zero. A query outside its
support still receives both predictions but is permanently masked from both
arms. Sixteen official query identities are matched to the 16 assimilation
centres using only frame-zero geometry and are permanently excluded from both
directions of Chamfer and from identity error.

## Ownership boundary

| Component | Ownership |
|---|---|
| Deform360 recordings, calibration, dynamic reconstruction pipeline, and material targets | Original Deform360 |
| PhysTwin simulator and released physical-model machinery | Original PhysTwin |
| Per-episode physical prediction used as the comparator | External physical backbone assembled from the original code |
| Sparse recursive discrepancy belief, continuation decision, and corrected nodal trajectory | Ours |
| Frozen arbitrary-point Gaussian displacement field | Ours |
| X0-only query artifacts, centre exclusion, direct-identity scorer, barriers, gates, and audit trail | Ours |

No MolmoMotion model prediction is used. MolmoMotion-Field supplied the research
idea of a shared queryable belief, while this experiment evaluates that idea as
an online correction layer for a physical deformable-object twin.

## Open-development operator decision

The fixed candidate grid contains three anchor counts (`64`, `128`, `256`), a
nearest-neighbour control, and the Cartesian product of three Gaussian
neighbour counts (`4`, `8`, `12`) with three length-scale fractions (`0.05`,
`0.10`, `0.20`). Candidate selection used only field-to-native nodal fidelity;
no future target score or target mask entered selection.

The selected candidate is
`gaussian-knn-normalized-v1-k04-length05pct`. Its equal-object, equal-anchor
field-to-native identity RMSE is `0.0004121574900429735 m`. The next two
candidates score `0.0004653549268697939 m` and
`0.0004820520074363731 m`.

As a descriptive result after target-free selection, the selected field beats
the physical comparator on both identity RMSE and symmetric Chamfer for every
anchor count and every one of the 27 open episodes (`81/81` selected-cell dual
wins). Across the complete grid it wins both metrics in all `810/810`
anchor-count/candidate/episode comparisons. Equal-object relative reductions
for the selected field are:

| Anchor count | Identity RMSE | Symmetric Chamfer |
|---:|---:|---:|
| 64 | 25.35% | 26.19% |
| 128 | 25.44% | 26.40% |
| 256 | 25.51% | 27.40% |

These are open-development results. They establish a stable hypothesis and do
not constitute confirmation evidence.

The immutable decision artifact is:

```text
gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/deform360-query-field-open27-v1-development/decision.json
SHA-256 110b3c1831898ff6b333f35236401761222f85eafac1dcbcea7b7183d5b434bd
```

## Rejected structured decoder

A matched ablation fits a proper Kabsch transform to each arm and frame and
interpolates only the residual deformation. It passes a deterministic pure
rigid-motion sanity check (maximum coordinate error
`1.1920928955078125e-7 m`, determinant in
`[0.9999999999999994, 1.0]`) but does not improve the real transfer.

Its target-free selected objective is `0.00041611100211358826 m`, which is
`0.96%` worse than total-displacement interpolation. Across the 27 matched
Gaussian anchor/configuration cells it is worse for target identity error in
`27/27`, worse for field-native
Chamfer in `27/27`, and improves target Chamfer in only `6/27`. V8 therefore
retains the simpler total-displacement field.

```text
gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/runs/deform360-rigid-residual-open27-v1-development/decision.json
SHA-256 b72faf6f7d4551622d6abbbd9521f05e46da7ef8cf4e9e17b161896889c7a2fa
```

## Why held v7 was withdrawn

Held v7 produced and sealed all 15 fresh physical and online predictions. The
first official target reconstruction also completed. Scoring then failed
before any metric was computed because the old evaluator required a complete
one-to-one assignment from every source node to an official identity. The
first case contained more source nodes than official identities, so such an
assignment was mathematically impossible.

This was an evaluator defect, not a negative model result. V7 has no completed
case score, aggregate, or gate decision. Its execution artifacts are not reused
by v8. The withdrawal report is immutable:

```text
gpuserver6000:/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v7/v7-outcome-withdrawal-report.json
SHA-256 7bcab7169fc2addad8e56b7bb5ca9086b5249e9a744e18b9d51a7f395098c1a3
```

After withdrawal, `002-rope-silk-ep0003` was used in a development-only
frame-zero query diagnostic. It is therefore retired from formal v8
calibration. The replacement is the previously untouched
`072-cotton-clohesline-ep0003`, frozen before media decoding. Public metadata
identifies it as a unimanual prehensile `drag center` episode, closely matching
the retired `drag middle` action. Its pinned metadata SHA-256 is
`11a1b742baf9bef68879ef076bbf0496381ff5906a6f0b7cd2a9581117595053`;
the source metadata census SHA-256 is
`7bf8897a4462fde7a6bf1a1fafab0fa213e7d41f5e2df72b6e002e7a7002fc33`.

The replacement preserves 15 calibration cases and the topology balance of
three filament, six sheet, and six volumetric cases. Per-object counts change
from five objects with three cases each to `002=2`, `072=1`, and
`083/085/092/170=3`. V8 therefore freezes and reports both equal-case and
equal-object aggregation.

## V8 information boundary

V8 uses two cohort-wide barriers:

1. Every fresh physical prediction, online prediction, and frozen-field
   manifest for the exact 15-case calibration cohort must be sealed before any
   official target reconstruction is authorized.
2. Every target is reconstructed, an artifact containing only official
   frame-zero identity IDs and positions is sealed, and both prediction arms
   are queried and sealed from those x0-only artifacts. All 15 queried
   prediction seals must validate before any future-target score capability is
   issued.

The scorer joins no source and target clouds. It consumes predictions already
queried in exact official identity order. Its single evaluation mask is

```text
shared field support
and not one of the 16 geometry-matched assimilation centres
and official future visibility
and official future validity
and finite official target coordinates.
```

The same mask is used for both arms and both Chamfer directions. There is no
arm-specific dropping, support repair, point-density weighting, or full
source-to-target assignment. Each case must have at least 90% frame-zero field
support and at least 32 hidden supported identities.

The frozen calibration gate requires:

- all 15 cases reported;
- at least 5% equal-case mean Chamfer improvement;
- aggregate identity RMSE improvement;
- at least 10 of 15 per-case Chamfer wins;
- no case with more than 10% Chamfer regression; and
- all support/cardinality requirements satisfied.

Only a calibration GO may authorize the unchanged six-case confirmation
cohort. Confirmation requires all six Chamfer wins, a one-sided sign-test
probability of `1/64`, at least 5% mean Chamfer improvement, identity
improvement, no large regression, and the same support requirements.

## Claim boundary

The published Deform360 benchmark evaluates open-loop per-episode,
multi-episode, and multi-object world models. This experiment instead receives
three sparse online material observations and evaluates a recursive correction
against its own sealed physical backbone. The metrics are expressed in the
same physical units, but the information sets, identity sets, time windows,
and aggregation are not the official Tables 3--5 protocol.

Consequently, even a successful v8 confirmation would support the claim

> sparse recursive observations can be transferred through one frozen shared
> field to improve a deformable physical twin at unobserved official material
> identities.

It would not by itself support “state of the art on Deform360.” A direct SOTA
claim requires a separate run under the official open-loop split and evaluator,
including the published particle-model baselines.
