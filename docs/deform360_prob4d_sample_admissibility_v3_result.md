# Deform360 Prob4D sample-admissibility v3 result

## Scope

This is a terminal source-preflight result on released Deform360 real-world
measurements. It requires neither new physical measurements nor human approval.
The frozen policy used provider support masks and released metric support masks,
but no predicted point values, prediction residuals, calibration outcomes,
future frames, confirmation payloads, or target outcomes.

The sole run is GitHub Actions run `31307498341`, attempt 1, at merged source
revision `0beaadab170e644fbaf3b4241d89d950e7a889ef`. Its compact artifact is
`deform360-prob4d-source-gate-31307498341-1`, artifact ID `9036556878`, with
GitHub artifact digest
`sha256:d9919442171b6eb0ad4967f88165dbcc1cf7365d0634b9a76fcca583c2b4d867`.
The frozen sample-admissibility policy ID is
`25c0a43b720accb3bacd16933774b3773a6bc951443b02b88498ca542d5fc51c`.

## Frozen outcome

| Stage | Outcome | Interpretation |
| --- | --- | --- |
| Public robot-gauge materialization | Passed | All 324 admitted streams were processed; 11 prior exclusions were retained |
| Complete-support gate | Passed | The frozen 313-stream candidate partition was materialized without replacement |
| Target-free sample admissibility | Failed | 102 streams were admissible, 211 were support-negative, and zero had technical failures |
| Source calibration | Not run | The preregistered admissibility gate blocked sample materialization before fitting |
| Automated source transfer gate | Not evaluated | No calibration or prediction-benefit decision exists |
| Independent confirmation | Not authorized | No confirmation payload or target outcome was opened |

The gate required at least 90% of the original 324-stream roster to remain
admissible, at least two streams per object, and all ten objects to remain
represented. Only 102/324 streams (31.48%) were admissible and nine of ten
objects retained at least two streams. The retained admissible-stream counts per
object, sorted without object identity, were
`0, 2, 3, 6, 9, 10, 10, 15, 17, 30`.

The failure is a source-support negative rather than a runtime failure. Every one
of the 211 rejected streams lacked the required eight independent spatial
clusters in at least one causal prefix window. Of those, 21 also lacked eight
metric-gauge correspondences and 14 additionally lacked 32 held-prefix point
rows. The dominant issue is therefore spatially concentrated support, which is
exactly the dense-correlation failure mode that the v3 policy was designed to
catch.

## Decision

This v3 run is terminal. Its thresholds will not be weakened and the run will
not be repeated. Calibration and confirmation remain closed because the frozen
source preflight did not authorize them. The result does not establish a
Prob4D prediction regression or gain: prediction point values and residuals were
never used, and the source transfer gate was never reached.

Prob4D remains an opt-in observation and covariance-calibration feeder in this
separate public-data experiment. It does not modify the frozen Causal4D physical
candidate, and this result supports no state-of-the-art claim. A later protocol
would need a genuinely new, source-independent way to obtain spatially redundant
causal prefix evidence; retuning this gate on the opened support result would not
be admissible.
