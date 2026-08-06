# TAPNext++ RGB-D Carrier Completion Source Result

Date: 2026-08-06

Status: the one-case source smoke passed; a separately frozen opened-cohort
transfer study is justified.

## Question

The frozen TAPNext++ carrier was accurate when strict calibrated triangulation
had support, but it retained only 52 of 76 eligible material point-frames. This
smoke asks whether the accepted multiview rows can calibrate a conservative
single-camera RGB-D fallback without using the manual target or the PhysTwin
state innovation.

The method preserves every strict row exactly. For each camera it compares
frame-zero-anchored RGB-D tracks with accepted carrier rows, removes a robust
constant offset for competence scoring, and divides median centered disagreement
by the square root of carrier-overlap fraction. One camera is selected. Only
that camera fills strict-carrier abstentions, so duplicating correlated cameras
cannot create extra precision. Added-row covariance contains local RGB-D
variance, carrier-disagreement covariance, and a 5 mm shared-bias floor.

## Result

Lower RMSE is better. The first query frame is not scored.

| Predictor | Supported rows | Support | Identity RMSE | Endpoint RMSE |
| --- | ---: | ---: | ---: | ---: |
| Exact persistence | 76/76 | 100.00% | 35.558 mm | -- |
| Strict TAPNext++ multiview carrier | 52/76 | 68.42% | 5.090 mm | 6.278 mm from the parent study |
| Completed carrier | 76/76 | 100.00% | **4.698 mm** | **5.374 mm** |

The 24 newly scored fallback rows have 3.709 mm RMSE. Relative to exact
persistence on the same 76 rows, the completed carrier improves RMSE by
86.79%. All four frozen gates pass: support, relative gain, overall RMSE, and
endpoint RMSE.

The target-free selector chose released camera 0. Before the manual target was
opened, its penalized agreement was 1.309 mm with 55/55 strict rows. The other
two cameras scored 1.763 mm with 20/55 overlap and 2.171 mm with 46/55 overlap.

## Interpretation

This closes the immediate support failure of the earlier TAPNext++ control.
Strict multiview geometry remains useful as a high-precision calibration
carrier even when it is too sparse to be the only observation path. A
single-view RGB-D fallback can recover the abstentions when camera competence
is established from those strict rows and camera correlation is not counted as
independent evidence.

It does not yet establish a Bayesian-PhysTwin improvement. The result scores
only an already-open prefix on one interaction; it does not inject the
observations into Warp, measure future Chamfer distance, test transfer across
cases, or establish calibration.

## Next Gate

Freeze the method and target-free physical-window rule on the remaining opened
PhysTwin source cases before running TAPNext++. First test provider transfer.
Only if that gate passes may the completed carrier enter a separately locked,
baseline-relative Bayesian state/discrepancy update. Future evaluation must
report both Chamfer distance and manual-track error and preserve exact fallback.

## Provenance

- implementation commit: `1ccd29a61a3289bfb408b03347316b450e8c8f81`
- prediction report SHA-256:
  `9c838a23b8ac9cf6c1e7a693d3a2633204e4b1d773bc3567b06d26ce37561ef6`
- prediction seal SHA-256:
  `da5c3bb7d81c61209516ae25192381b0ef0cc2a1fe00e8816b0c9fa3be18af20`
- result file SHA-256:
  `32b97beaa9583829b353d85b74aead4be5d93e3edc6f86f238157a6af5837dfa`
- canonical result SHA-256:
  `5bfbce05f59e7742a0b12ca99c29a8f0d90a3ba33ac55127ebf3ae1566136c23`
- focused source verification: 5 tests passed on `gpuserver4090`

No held-v8 runtime, target, query, score, barrier, or outcome artifact was read
or modified.
