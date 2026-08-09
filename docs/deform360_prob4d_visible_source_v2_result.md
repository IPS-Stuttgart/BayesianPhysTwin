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

## Independent audit

GitHub-hosted audit run `31303263941` independently revalidated the compact
artifact at merged validator revision
`dbeab4f5a5c8279b82404b3dc911c39ddccab10d`. The audit completed without
dataset, runner-root, confirmation, target-outcome, or held-protocol access. Its
artifact is `deform360-prob4d-visible-source-v2-independent-audit-31301431579`,
artifact ID `9035153180`, with GitHub artifact digest
`sha256:f3da9ff4000d89cd1396761e41c65c566b771263d097603495280c52847103e3`.

The validated audit ID is
`86d6998941fe76769a007494c1ff84ac820ada96f414504172cd2daacba26511`.
The committed receipt has SHA-256
`0797061b7874b20c1d37eec3cb671897ac46fefac8e639545149e5c0cbca2d44` and
records `public_released_measurements_used=true`,
`new_measurements_required=false`, `human_approval_required=false`,
`source_gate_evaluated=false`, and `confirmation_access_authorized=false`.

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
