# Deform360 Prob4D target-free-visible source v2 result

## Scope

This is a public-real-world-data source-pipeline result. It uses released
Deform360 measurements; it requires neither new physical measurements nor human
approval. The only advancement criterion is the preregistered automated source
gate. Confirmation payloads and target outcomes remained closed.

The frozen run is GitHub Actions run `31301431579`, attempt 1, at source revision
`136f72b996e9c76b0bab3ab5db5d0fe7172e0307`. Its compact artifact is
`deform360-prob4d-source-gate-31301431579-1`, artifact ID `9034737368`, with
GitHub artifact digest
`sha256:caa8d5ea887ec5273c306dd8de59d57056181ef139c98ac7acb76185032a3828`.

## Frozen outcome

| Stage | Outcome | Interpretation |
| --- | --- | --- |
| Target-free metric batch | Passed | 313 supported streams and 11 retained visibility exclusions across all 10 objects |
| Target-free support gate | Passed | The exact v2 camera-eligibility contract admitted the retained partition without replacement |
| Source-sample materialization | Technical failure | The pipeline stopped before calibration or residual scoring |
| Automated source gate | Not evaluated | No transfer or calibration decision exists |
| Independent confirmation | Not authorized | No confirmation payload or target outcome was opened |

The compact artifact records the sample-stage standard-error digest as
`5da90e87f5cd814b48200f7a978309a634b7d2a5adddf6aefec1a045ac4e5b7d`, but
deliberately omits the error text. Therefore this record does not claim a
specific exception or causal diagnosis. A target-free post-open diagnostic found
one supported stream with 31 projected points, but that observation is not
evidence that the stream caused the materialization failure.

## Claim boundary

This is a source-pipeline technical terminal, not a negative or positive model
result. It establishes no prediction gain, calibration result, confirmation, or
state-of-the-art claim. Prob4D remains an opt-in observation feeder in this
separate experiment and does not alter the frozen Causal4D physical candidate.

The v2 run is terminal and will not be rerun or weakened. Any future v3 must be a
newly versioned protocol that preregisters stronger target-free agreement between
metric support and sample admissibility, plus a sanitized compact failure class,
before source residuals are opened. Such a protocol would still use public
real-world measurements and would still require no human approval.
