# Guarded MatPhys surface-UQ source protocol v2

## Question

This source-only study asks whether target-excluded disagreement among official
MatPhys material proposals contains useful uncertainty information when it is
admitted by a target-free replay-quality guard and the point prediction itself
remains the frozen DEFORM result. It does not ask whether MatPhys replaces
DEFORM as the best mean predictor.

The candidate and every comparator therefore use the exact same `prediction_m`
bytes. Only the predictive covariance changes. This protects the existing
DEFORM result and makes an NLL, coverage, or selective-risk improvement
attributable to uncertainty rather than to a hidden mean change.

## Source denominator

The complete denominator is the ten previously opened Deform360 development
cases in
[`matphys_surface_uq_source_v2.json`](../configs/sota/matphys_surface_uq_source_v2.json).
No case may be replaced. Ordinary predictions, exact-mean fallbacks, retained
technical failures, and unscorable outcomes are reported separately.

These cases are outside the eleven-object MatPhys training universe. Every
MatPhys checkpoint may therefore contribute without training on the evaluated
physical object. The ensemble is replayed through the official PhysTwin Warp
backend, and repeated same-field replays measure numerical simulator noise
separately from between-checkpoint spread.

## Observation boundary

Camera names alone define two disjoint panels before source suffix decoding.
Frame-zero DINO part features use the lexical complement of the twelve scoring
cameras. Future source surfaces are reconstructed only from the scoring panel.
No RGB, mask, depth, Splatfacto, or tracking artifact is shared between the two
panels.

The source endpoint contains frames 58 through 75 of the 76-frame prediction
window already bound by the sealed DEFORM prediction. An initial source-only
attempt requested five additional unscored tail frames and stopped before
reconstruction because the raw robot stream was not readable outside its CI
account. A second preflight established that the derived known-action carrier
was CI-private too; it stopped before writing an artifact and before any source
reconstruction or metric was available.

The prospectively amended scoring reconstruction therefore consumes only the
registered scoring-panel RGB videos, camera calibration, and source-fitted SAM2
object masks. Deform360's released depth stage supports this input without a
robot-state carrier. The runner does not invoke URDF rendering, so gripper
pixels are not explicitly excluded from the reconstructed surface. This is a
declared limitation of the custom outcome rather than evidence about the mean
or covariance model. The five unscored tail frames remain omitted, and neither
the prediction bytes nor any advancement threshold changes. This remains a
custom disjoint-camera Deform360 reconstruction, not an official processed
annotation or official benchmark score.

The retained failed-attempt manifest is SHA-256
`103ff5eb0768a454c7ef242ee860e94f70bb67be9381f89e331ababc3f3afe1c`.
The first robot-independent source execution then decoded the scoring-camera
suffix and produced SAM2 masks, but stopped during Splatfacto initialization:
the inherited Python 3.12 environment contained a source-only `gsplat` package
without its CUDA backend. Its retained manifest is SHA-256
`57a4df4a7824b632ee40c7706a4e049aa7a5d2965e530c8c2222f6f232b65918`.
No reconstructed endpoint or source metric existed when the runtime amendment
was frozen, and no prediction or advancement threshold changed.

The endpoint runner now validates the scoring runtime before any source suffix
decode. The frozen identity is Python 3.10.20, NumPy 1.26.4, Torch
2.4.0+cu121 with CUDA 12.1 on compute capability 8.9, torchvision
0.19.0+cu121, precompiled `gsplat` 1.4.0+pt24cu121 with
`CameraModelType`, Nerfstudio 1.1.5 with Splatfacto and Gaussian export, and
OpenCV 4.10.0.84. A mismatch is a retained technical failure, not permission
to substitute another runtime or source case.

The first pinned-runtime execution passed that preflight but retained a failure
before Splatfacto because fewer than three scoring cameras produced masks. Its
manifest is SHA-256
`ffc8d5e23a730fe49c6e3b51da8f085ed0d1db36e7cbd95883710c10a015dd0b`.
The runner had caught each per-camera exception but omitted those records from
the top-level failure artifact. A fourth prospective operational amendment
adds only the failure stage, per-camera diagnostics, and successful-mask count
to retained failures. It does not change SAM2, the camera panel, the three-view
support threshold, the runtime, any prediction, or any advancement gate.

That diagnostic rerun showed the same source-independent error on all twelve
cameras: the clean Python 3.10 environment lacked the `decord` video reader, so
SAM2 never loaded a frame or inferred a mask. The retained manifest is SHA-256
`64c6fd06f73c2857bfdaccde30830abbab5e8027302102fb8e3ba2e9ab1da355`.
The fifth operational amendment adds and verifies only `decord==0.6.0`, matching
the existing successful SAM2 environment. No mask, reconstructed endpoint, or
source metric was available, and all scientific settings remain unchanged.

The complete eligible-panel Warp replay then exposed a separate runtime parity
defect before any scoring-camera suffix was decoded for the affected case. The
registered official trajectories were generated with Warp 1.16.0, but the
auxiliary ensemble process inherited Warp 1.15.0. On a nearly deterministic
case, a 0.293 mm reference difference divided by a 0.003 mm replay floor and
failed the frozen relative parity check. The retained manifest is SHA-256
`a199425fb5754b65940d1ca500944e1536913003803a9e7bcfd9bcf35e079002`.

The sixth operational amendment fixes the mismatch rather than relaxing the
parity threshold: Warp 1.16.0 is now a fail-closed runtime identity, and all
eight eligible source covariances must be regenerated under that version. The
earlier 1.15.0 single-case smoke metric is exploratory and cannot enter the
final source gate. The DEFORM mean bytes and every advancement threshold remain
unchanged.

The version correction did not remove the historical trajectory difference.
An exact rerun of the original official producer, now in its pinned Python
3.10, Torch 2.4.0, CUDA 12.1, and Warp 1.16 runtime, differed from the stored
trajectory by 0.290 mm RMSE. The MatPhys wrapper, however, agreed with that
fresh official rerun to below 0.001 mm. Two further independent official
replays clustered within 0.007 mm of the first. This falsifies the narrower
claim that the adapter was wrong and exposes a historical-to-current replay
provenance drift instead.

The seventh amendment therefore distinguishes two quantities. Adapter parity
is fail-closed against a separately constructed current official replay made
by the exact registered producer. The difference between that replay and the
historical trajectory is reported separately as a provenance diagnostic. It
is included conservatively when deciding whether checkpoint disagreement rises
above the execution floor, but it is not relabeled as material uncertainty and
is not added to the predictive covariance. The prediction mean and source gate
remain unchanged.

## Target-free uncertainty guard

The original v1 arm stopped before additional source reconstruction because
one of eight replayable cases failed the frozen signal-over-floor test. The v2
arm does not weaken that test. It treats the failed replay artifact as a
target-free abstention: quality-passing cases use the MatPhys covariance, while
an abstaining case uses exactly the leave-one-case-out isotropic comparator
covariance. In either branch, the mean is the same byte-identical DEFORM array.

The guard is fixed before opening the abstaining case's scoring-camera suffix.
It reads only the sealed current-reference parity result, within-replay spread,
historical provenance drift, and between-checkpoint spread. It does not read a
surface residual, NLL, coverage, or future point metric. An isotropic fallback
is a tie, not a MatPhys win, in the case-level win count.

## Estimand and comparators

For each future frame, scoring-camera depth is backprojected into the world
frame. Each registered DEFORM mean node is associated with its nearest surface
point, subject to the frozen 50 mm metric limit. The event residual is a 3-D
point-to-surface displacement in metres.

When admitted, the MatPhys covariance is the target-excluded between-model plus
within-replay covariance for the same graph node. A scalar multiplier is fit
on the other admitted source cases, with a fixed 5 mm
observation/reconstruction floor. The two comparators are:

1. a leave-one-case-out isotropic Gaussian around the identical DEFORM mean;
2. a leave-one-case-out radial split-conformal sphere around that same mean.

Metrics are first averaged within each case and then equally across cases.
Coordinates, nodes, cameras, and frames are not treated as independent
experimental replicates.

## Advancement rule

The guarded covariance advances only if at least eight cases are scorable,
each scored case retains at least half of its attempted events, and all
registered gates pass: at least six strict case-level NLL wins against the
isotropic comparator, at least 0.05 nats/event equal-case NLL improvement, 90%
coverage between 80% and 98%, and at least 5% lower mean 90% ellipsoid volume
than the conformal sphere. An isotropic fallback contributes a tie to the win
count. A source failure leaves DEFORM unchanged and forbids a fresh target run.

Passing this source gate authorizes only a separately locked, genuinely fresh
public-object calibration evaluation. It does not authorize held-v8, DLO4,
DLO5, any revoked covariance cohort, or a SOTA claim.

## Status

The v2 guard is locked before any additional scoring-camera reconstruction.
The predecessor v1 result remains immutable at
[`replay_quality_result.json`](../results/sota/matphys_surface_uq_source_v1/replay_quality_result.json).
No source calibration result or fresh-target evaluation is claimed yet.
