# Deform360 Bias-Aware Prospective V1 Result

## Status

The frozen prospective protocol stopped at a target-free calibration-support
rejection. All nine calibration cases received exactly one sealed disposition
before any calibration future was opened: five predictions and four
no-replacement quality failures. No target case was staged, decoded, or scored.

This is not an accuracy result for the bias-aware guarded update. It establishes
that the complete automatic pipeline did not provide enough evaluable fresh
objects to run the locked accuracy and non-regression gate.

## Calibration Cohort

| Stratum | Locked objects | Sealed predictions | Quality failures | Required evaluable |
| --- | ---: | ---: | ---: | ---: |
| Filament | 3 | 1 | 2 | 2 |
| Sheet | 3 | 2 | 1 | 2 |
| Volumetric | 3 | 2 | 1 | 2 |
| **Total** | **9** | **5** | **4** | **7** |

All four failures occurred at the frozen frame-zero reconstruction admission:

- `160-hose`, episode 1;
- `174-chain`, episode 1;
- `015-airbag-cloth`, episode 6;
- `100-puppet`, episode 9.

Each frame-zero fit completed, but the masked point cloud failed the locked
minimum-point and finite-value check after low-opacity Gaussian filtering. No
threshold was changed and no object or episode was replaced.

## Gate Decision

The complete prediction cohort has result SHA-256
`581bfb211d312d0cadfdd5d1a0005d4cba1369f0eb70015438371a07a0503dc3`.
Once sealed, the following upper bounds were fixed regardless of any unseen
outcome:

| Locked support condition | Maximum available | Requirement | Result |
| --- | ---: | ---: | --- |
| Evaluable calibration objects | 5 | at least 7 | **Fail** |
| Evaluable objects per stratum | 1 / 2 / 2 | at least 2 each | **Fail** |
| New eligible groups | 5 | at least 5 | Still possible |
| Combined eligible groups | 9 | at least 9 | Still possible |
| Exact finite-sample coverage | 90% | at least 90% | Still possible |

Because the first two failures are irreversible, outcome values cannot make the
gate pass. The non-authorizing support-rejection artifact has result SHA-256
`eeee3907be8404935e32a43c9d11e5d2ffb29f2af0a7d0af3b0faa92429771e0`.
It records `calibration_gate_passed=false` and
`target_access_authorized=false`. Performance-dependent gates were not
evaluated.

## Interpretation

The source-v4 improvement remains an open-development result. This prospective
study neither confirms nor falsifies its fresh-object accuracy because hidden
future trajectories were never read. It does falsify the assumption that the
current frame-zero automatic twin path has sufficient coverage for this locked
three-stratum protocol. The concentration of two failures among three filament
objects is especially important for a method intended to generalize across
deformable families.

This distinction matters for the paper. Reporting only the five successful
cases would condition on reconstruction success after selection and violate the
locked minimum-support gate. Opening their outcomes could not rescue the claim
and would spend independent data without changing the decision.

## Research Decision

Do not tune the guarded update, reconstruction threshold, or selected cohort on
these cases, and do not open the twelve reserved target objects. The next source
milestone is an outcome-blind automatic-twin coverage study on already-open
objects. It should compare the current Gaussian-splat export with a predeclared
robust frame-zero initializer, such as multiview silhouette or depth-supported
surface construction, while preserving the same physical-backbone interface
and exact persistence fallback.

A replacement initializer should earn promotion by target-free criteria:

1. higher object- and stratum-level admission on open source objects;
2. valid metric geometry, support, and graph connectivity;
3. byte-identical behavior when the original initializer is accepted;
4. no use of future RGB, hidden tracks, or outcome metrics;
5. a new lock before selecting or decoding another fresh cohort.

Only after that coverage gate passes is another prospective accuracy study
justified. Until then, Bayesian-PhysTwin has a promising source result and a
clean prospective pipeline-coverage failure, not a state-of-the-art result.

## Evidence

- `results/sota/deform360_bias_aware_guarded_belief_prospective_v1/calibration_prediction_cohort_seal.json`
- `results/sota/deform360_bias_aware_guarded_belief_prospective_v1/calibration_support_rejection.json`
- `results/sota/deform360_bias_aware_guarded_belief_prospective_v1/download_manifest.json`
- `results/sota/deform360_bias_aware_guarded_belief_prospective_v1/prediction_seals/`
- `results/sota/deform360_bias_aware_guarded_belief_prospective_v1/prediction_reports/`
- `results/sota/deform360_bias_aware_guarded_belief_prospective_v1/quality_failures/`

The checksum-bound arrays and complete logs remain on `gpuserver6000` at
`/mnt/corsair/florianpfaff/deform360-bias-aware-prospective-v1`.
