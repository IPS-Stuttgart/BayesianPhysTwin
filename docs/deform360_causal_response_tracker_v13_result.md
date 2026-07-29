# Causal-Response Tracker V13 Source Result

## Decision

The frozen source competence gate **failed**. Stop the V13 fixed-identity
TAPNext++ carrier and do not construct either a state update or a readout
update from these observations.

This is a post-open source result on eight previously examined Deform360
cases. It is not a future-prediction, transfer, calibration, confirmation, or
state-of-the-art result.

## Registered Result

| Quantity | Result |
| --- | ---: |
| Locked cases | 8 |
| Sealed tracker predictions | 6 |
| Exact V13 query abstentions | 2 |
| Scheduled identities | 96 |
| Supported identities | 5 |
| Pooled endpoint support | 5.21% |
| Cases with at least 50% support | 0/6 |
| Scored cases | 1/6 |
| Provider wins | 1/1 scored, 1/6 registered |
| Provider endpoint RMSE | 0.527 mm |
| Exact-persistence endpoint RMSE | 0.635 mm |
| Relative endpoint gain | 17.10% |
| Provider late-prefix RMSE | 0.535 mm |
| Exact-persistence late-prefix RMSE | 0.636 mm |
| Mean accepted panel disagreement | 2.223 mm |
| Raw nominal 90% coverage | 100% on 5 rows |

Only `061-cup` retained any jointly accepted endpoint identities: 5 of 16.
The other five tracker cases retained zero identities after independent
proposal/validation corroboration. Proposal-only or validation-only support
was also sparse outside `061-cup`.

The low RMSE, positive 17.10% gain, 2.223 mm panel agreement, and raw 100%
coverage therefore do not rescue the method. They describe five rows from one
low-motion case. The mean NEES was 0.0196, so the uncalibrated covariance was
very broad relative to those errors; it is not evidence of calibrated
uncertainty.

The registered gate passed only:

- exactly six sealed provider predictions;
- at least 10% gain over persistence on the scored subset;
- the mean cross-panel disagreement threshold.

It failed pooled support, per-case support, minimum scored cases, provider
wins, and both object-balanced accuracy gates, whose minimum-case conditions
were not met.

## Evidence Order

All six provider predictions and both exact query abstentions were sealed
before the released prefix identities were deserialized. The evaluator then
wrote the prediction-completeness barrier and opened only the manual
identities through frame 57.

No object observation or metric after frame 57 was read. No state or readout
update was constructed. V1 sealed targets and all held-v8 artifacts and
processes remained untouched.

The first runtime attempt stopped before camera decoding because its selected
environment lacked OpenCV. It produced no provider artifact. Attempt 2 used an
existing GPU environment containing the same pinned TAPNext++ code and
checkpoint plus the required camera dependencies; the code, protocol, cases,
inputs, queries, covariance rules, and gates were unchanged. Attempt 1 remains
preserved as a technical failure.

## Interpretation

V13 repaired a frame-zero admission problem:

- 6/8 carriers were admitted;
- two strict 3+3 and four inflated 2+2 arms materialized;
- no query-construction failure occurred.

The tracker result identifies the next bottleneck. Frame-zero multiview support
does not imply that a fixed material identity remains observable, triangulable,
and independently corroborated throughout the action prefix. Requiring both
panels correctly prevented unsupported camera tracks from becoming a
high-confidence Bayesian update, but left only 5/96 usable identities.

This does not show that TAPNext++ is inaccurate whenever it is supported. The
one supported case was accurate and improved over persistence. It shows that
this fixed frame-zero, camera-only interface is not a deployable observation
provider across the registered source panel.

## Consequence

Do not:

- weaken the two-panel support or covariance rules;
- tune cameras, queries, thresholds, or TAPNext++ on these opened cases;
- add tactile event timing to this failed carrier;
- construct a bias-aware update from its five supported rows;
- use the 17.10% subset gain as Bayesian-PhysTwin or SOTA evidence.

The source result supports exact fallback and closes this V13 route. A future
observation provider would need genuinely new information, such as an
independent modality or an independently validated dynamic identity mechanism,
and a fresh preregistered source cohort. Fresh Deform360 selection remains
blocked until the independent held-v8 all-attempt hash-only exclusion manifest
exists.

## Provenance

- Implementation commit: `d4f43b24117b68ebd611fd97cc1d1feb68b30d14`
- Protocol commit: `377f3426f6cedd347cfd4ee42e40f40b4e208acf`
- Protocol SHA-256:
  `f716bcf90bc09fe44727335eb98226353c5e0ef3e5b920ed83ef5943621918c3`
- Parent V13 feasibility result:
  `63a9ffde07de4e0378605b8574a8d4a8acfe6a382c65df2be185c47f6276239c`
- Prediction barrier SHA-256:
  `d796a315568de16ec0af938903f9d50aa17f016731f356d87014ec6e6eb1cba3`
- Competence result SHA-256:
  `9ca4e1f6b4c9d5d5e12d019e3ad9fe29cec6211b57799ee54b918095f719abe1`
- Canonical result digest:
  `cc8752943761a03c0992a47f65d150102e5b896cbc26284399a13c316b26c744`
- Frozen Linux verification: 1,499 passed, 28 skipped; changed-file Ruff
  clean.

The compact barrier and result are archived under
`results/sota/diagnostics/deform360_causal_response_tracker_v13_source/`.

