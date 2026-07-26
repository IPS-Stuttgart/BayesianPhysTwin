# Covariance-aware sparse identity smoke v1

## Status

The locked one-case development smoke failed. The exact sparse-identity arm is
stopped and must not be expanded to the opened 19-case cohort or tuned against
this future.

This is a post-open diagnostic on the previously examined
`single_lift_cloth` interaction. It is not independent evidence, a calibration
result, or a state-of-the-art claim.

## Locked comparison

The candidate retained the released `final_data` pseudo-tracks as its dense
geometry channel. CoTracker3 contributed only a sparse material-identity
endpoint update with:

- residual-independent perception reliability;
- duplicate-camera collapse;
- conservative unknown-correlation ray information;
- an explicit shared-bias covariance floor;
- inflated uncertainty for two-view fallback;
- metric observation covariance in square metres; and
- one robust innovation likelihood.

The primary prediction arm,
`causal_selected_dense_relative_cap_temporal`, and all settings were committed
in `configs/sota/phystwin_sparse_identity_smoke_v1.json` before the candidate
future was produced. A locked analyzer suppressed every non-primary arm and
metric oracle from the broad diagnostic runner.

## Result

| Predictor | Future CD | Future manual-track error |
|---|---:|---:|
| Raw physical replay | 24.625 mm | 60.098 mm |
| Frozen released-dense arm | **11.963 mm** | **52.976 mm** |
| Released dense + covariance-aware sparse identity | 12.146 mm | 61.215 mm |

Relative to the frozen released-dense comparator, the sparse identity update:

- worsens CD by **1.53%**;
- worsens manual-track error by **15.55%**; and
- worsens track error by **1.86%** even relative to the raw physical replay.

It therefore fails the locked 5% track-improvement gate, the 1% CD-regression
tolerance, and physical-baseline non-regression.

The failure is not caused by absent automatic observations:

| Observation diagnostic | Value |
|---|---:|
| Identities with some support | 2,920 |
| Valid identity point-frames | 21.83% |
| Valid observations using two-view fallback | 63.57% |
| Median observation standard deviation | 22.97 mm |
| Mean prior reliability | 0.000228 |
| Innovation uses PhysTwin residual in prior reliability | No |
| Robust innovation likelihood count | 1 |

## Interpretation

The implementation satisfies the intended causal and covariance boundaries,
but calibrated covariance does not make a biased camera-derived point estimate
beneficial. Sparse multiview identities can be plentiful and geometrically
coherent while still degrading material-identity prediction. This agrees with
the Prob4D finding that uncertainty is useful for honest consistency and gauge
fusion, not as an automatic point-estimate improvement.

The 63.57% two-view share and 22.97 mm median uncertainty are also consistent
with the broader Deform360 evidence: camera redundancy can be weak, and
camera-internal agreement cannot identify coherent common-mode bias. Repeating
or loosening this camera-only update on the same opened cases would not answer
that identifiability problem.

The next credible automatic identity channel needs independent information
that can distinguish object motion from camera bias, such as source-calibrated
physical/action support, held-out depth, tactile contact, or another independent
modality, together with exact baseline fallback. It should be gated on
object-disjoint source data before any fresh-object evaluation.

## Reproducibility

- implementation base: `abb897e8808e9897af47232ee265f2111631fc91`
- locked protocol/analyzer commit: `02952940c002d69e2712d691ae7c7790fc06ce3e`
- protocol SHA-256:
  `108fff7e5e38a01cfbc232ccfe8c4dfce4ce7668fab9230cef92651a94ef4394`
- sealed broad candidate SHA-256:
  `7c38b67195e57035bd405b077498e3a8fa0e16a15f9872fac9ea802e80d27c01`
- compact result SHA-256:
  `aeaee4f0e94e5d9a64070b4e94b52a6bd68987ead0f36178883d6dadb6441c09`
- remote root:
  `/mnt/corsair/florianpfaff/bpt-sparse-identity-smoke-v1`

The first runner attempt failed before loading a trajectory because a NumPy
1.26 environment could not deserialize the NumPy 2 pickle namespace. Its log
was retained. The identical frozen command then ran with the validated NumPy 2
runtime; stderr was empty.

The exact locked checkout passed 863 tests in 12.05 seconds on
`gpuserver6000`, with CUDA hidden. Ruff passes on all files changed for this
milestone. A repository-wide Ruff scan still reports eight pre-existing issues
in unrelated files; they were not modified.

The compact permitted result is archived at
`results/sota/diagnostics/phystwin_sparse_identity_smoke_v1/result.json`.
