# PhysTwin recursive gauge-RBF source smoke v1 result

Date: 2026-07-26

Status: the locked one-case source smoke failed safely. This exact
camera-only arm is stopped and must not be expanded to the opened 19-case
cohort or tuned against this future.

## Question

The smoke tested whether strict three-view CoTracker3 material observations
could improve the frozen released-dense PhysTwin comparator when a recursive
update:

- carried metric observation covariance;
- represented shared camera translation as an explicit nuisance;
- propagated a full-covariance global-plus-local RBF belief through
  action-conditioned physical rotations; and
- returned the comparator byte-for-byte unless an independent prefix
  point-cloud gate improved by at least 1%.

The predictor could use RGB and released pseudo-track geometry only through the
allowed prefix. It could not receive future RGB, future object observations, or
manual material tracks.

## Prefix decision

The recursive filter accepted four internally valid updates at frames 1, 19,
38, and 59. Their observation counts were 15, 12, 7, and 4, respectively.
Nevertheless, the resulting trajectory provided effectively no independent
prefix improvement:

| Prefix arm | One-sided Chamfer distance |
| --- | ---: |
| Frozen dense comparator | 11.304499 mm |
| Recursive candidate | 11.304498 mm |
| Relative improvement | 0.000014% |
| Required improvement | 1.000000% |

The prefix gate therefore rejected the update and emitted the frozen dense
comparator exactly. This distinguishes an estimator update from evidence that
the update is useful: passing an internal likelihood or identifiability check
does not establish baseline-relative predictive value.

## Future result

Because the prefix gate rejected the arm, candidate and comparator future
metrics are identical:

| Arm | Future CD | Future manual-track error |
| --- | ---: | ---: |
| Raw physical trajectory | 24.625 mm | 60.098 mm |
| Frozen dense comparator | 12.068 mm | 52.902 mm |
| Recursive candidate after exact fallback | 12.068 mm | 52.902 mm |
| Candidate change vs comparator | 0.00% | 0.00% |

The early, middle, and late horizon metrics are likewise identical. The
preregistered requirement of at least 5% future manual-track improvement was
not met. No recursive correction covariance is reported as predictive
uncertainty because the selected output contains no admitted correction.

## Why this arm stopped

The target-free prefix audit already showed a difficult observation regime:

- only 7.92% of identity-frame entries had strict support from all three
  cameras;
- the median metric observation standard deviation was 15.685 mm;
- mean residual-independent prior reliability was 0.000252; and
- support for at least four selected centers ended at frame 59, well before
  the fit and prediction boundaries.

The recursive estimator handled this evidence conservatively and the outer
gate prevented a regression. The result does not show that recursive Bayesian
state estimation is ineffective. It shows that this fixed strict-three-view
camera feeder supplied no measurable baseline-relative support for its update
on the opened development case.

Together with the completed automatic CoTracker3 open-22 arm and prior
camera-only virtual-sensing failures, this result does not justify another
threshold, rank, covariance, or support search on the same observations. A new
arm needs evidence that changes the identifiability problem, such as:

- an independent sparse depth, tactile, or registered actuator/contact
  modality;
- a source-calibrated action-supported admission rule with measurable
  headroom over the unchanged physical baseline; or
- a genuinely stronger material-identity observation artifact evaluated under
  a new object-disjoint source protocol.

The recursive gauge-aware belief implementation remains a tested reusable
component for such an arm. The rejected camera feeder is not promoted.

## Information boundary

- This was one previously opened development interaction, not independent
  confirmation or a state-of-the-art comparison.
- Future RGB and future object observations did not form the prediction.
- Manual tracks were loaded only by the scoring stage after prediction
  sealing.
- The prefix admission metric used released pseudo-track geometry, never
  manual identities.
- No held-v8 target, query, score, barrier, outcome, or process was accessed.
- The opened future must not be used to tune or retry this exact arm.

## Provenance

- implementation commit:
  `cc1dbf1f2b8cf48b1724e33b3b8ae63d1a0f2ef5`
- protocol commit:
  `adac0a6`
- locked protocol SHA-256:
  `ae5fa97a80611edbf48e8c04ea98208c84ebf9cbcc2c9fb5fffad9801863c751`
- prefix artifact SHA-256:
  `8190604b52804feb73cf587f68ade93a42e2256a75f72840c9a3a339b7abfaa7`
- prediction NPZ SHA-256:
  `f215e38768eb3642bf697c52fe7fa709a4dd43cb113cc9f87ef577bdefc8e4c5`
- prediction manifest SHA-256:
  `374de5b0d7e2cd42299869cc47d1324085d7fe742d2c01acd9585cbac5db51d5`
- result SHA-256:
  `a8f7c75675495e0e7fda4d65c31b2dce00ca42d028f20e2627b3ef1598834704`
- compact evidence:
  `results/sota/diagnostics/phystwin_recursive_gauge_rbf_source_v1/`
