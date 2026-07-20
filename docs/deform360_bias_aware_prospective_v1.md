# Deform360 Bias-Aware Prospective V1

Status: locked before selected-object download or media access.

This protocol is the independent public-data test of the source-frozen
bias-aware guarded belief update. Its executable lock is
`configs/sota/deform360_bias_aware_guarded_belief_prospective_v1.json`, with
canonical SHA256
`b6b19be5eaadf830a77f36cccddd38f5b7a35527ca21f7743d2ef147fceabbce`.

## Claim Boundary

The experiment tests whether the frozen online correction improves the exact
selected raw/physical backbone on fresh objects while preserving bit-exact
fallback. It is not official Deform360 Table-4 parity. Even a passing result
does not by itself establish general state of the art, a universal camera-only
safety theorem, or calibrated material parameters.

The method is bound to Bayesian-PhysTwin commit
`06f75a4406289384228a988df959e3c2af44510e`, source summary SHA256
`dbad5fd3b4d572d515d38b9bb31df84a2f036c223aaed3aa0810c25fbec3e015`,
and source lock SHA256
`5f5672d35aa41e276f1dd5ace54b6694b0139ff2a562e3c3a24558fa555c9dd6`.
No threshold, basis rank, reliability rule, covariance model, observation
pipeline, or method family may change after this lock.

## Metadata-Only Selection

The dataset is pinned to revision
`7fea8e20231a47641d1d2bc8791920ec4e62ec5e`. Candidate pools were declared
from top-level object directory names only. No selected-object image, video,
geometry, tactile stream, action metadata, or score was accessed. Objects are
ranked within each name-only stratum by
`SHA256(seed:object:stratum:object_id)`. Episodes are ranked separately by
`SHA256(seed:episode:role:object_id:episode_id)`.

All 40 objects used or reserved by the earlier source, replication, and
selective-virtual-sensing protocols are excluded. Before lock, path-name-only
audits on both `gpuserver6000` and `gpuserver4090` found zero paths matching any
of the 21 newly selected objects.

### Calibration cohort

| Stratum | Object | Episode |
| --- | --- | ---: |
| Filament | `160-hose` | 1 |
| Filament | `174-chain` | 1 |
| Filament | `076-rubber-bands` | 0 |
| Sheet | `175-plastic-bag-cloth` | 3 |
| Sheet | `011-green-cloth` | 0 |
| Sheet | `015-airbag-cloth` | 6 |
| Volumetric | `163-bear` | 1 |
| Volumetric | `100-puppet` | 9 |
| Volumetric | `168-cat-big` | 0 |

### Target cohort

| Stratum | Object | Episodes |
| --- | --- | --- |
| Filament | `075-leather` | 3, 1 |
| Filament | `123-pipe-cleaner` | 4, 7 |
| Filament | `080-wool` | 4, 2 |
| Filament | `143-silicone-wristband` | 9, 7 |
| Sheet | `165-glove-yellow-cloth` | 1, 9 |
| Sheet | `066-glove-half-black-cloth` | 0, 1 |
| Sheet | `112-wristband-cloth` | 0, 6 |
| Sheet | `091-net-cloth` | 8, 4 |
| Volumetric | `139-rubber-ball` | 3, 0 |
| Volumetric | `121-croissant-plush` | 7, 1 |
| Volumetric | `120-bread-plush` | 6, 8 |
| Volumetric | `164-sheep` | 5, 1 |

Failed objects or episodes are never replaced.

## Calibration Gate

Calibration is a prospective test of the already frozen selector, not a new
development loop. The only permitted operation is refitting the direct
source-group regret bound after every calibration prediction has been sealed.
The candidate construction, physical-agreement threshold of 0.40, and required
improvement of 0.005 mm remain unchanged.

The target phase is admitted only when all of the following hold:

1. at least seven of nine calibration objects and at least two per stratum are
   evaluable;
2. at least five calibration objects contain a target-free eligible update;
3. together with the four inherited source groups, at least nine eligible
   object groups calibrate the bound;
4. the exact finite-sample level is at least 90%;
5. the combined upper regret bound remains below `-0.005 mm`;
6. both calibration object-balanced primary regrets are negative;
7. no accepted calibration object is harmful on either primary metric;
8. every rejection is the exact baseline.

No method change is allowed if this gate fails. The failure is reported and all
target futures remain sealed.

## Target Evaluation

Each update reads exactly RGB frames `[0,u]` for `u` in `[19, 38, 57]`.
Future-only frames 20--37, 39--56, and 58--75 are scored. The co-primary
metrics are hidden identity RMSE and hidden symmetric Chamfer, averaged within
episode and then physical object. Point coordinates and frames are not treated
as independent samples.

A positive target claim requires:

- at least nine evaluable objects and three per stratum;
- negative object-balanced differences for both primary metrics;
- object-cluster upper 95% bounds below zero for both metrics;
- no mean regression in any stratum;
- at most 10% harmful accepted objects;
- bit-exact fallback for every rejected update;
- complete reporting of quality failures without replacement.

## Information Order

```text
commit method, protocol, and runner
-> download only the 21 locked objects
-> stage and seal all calibration predictions
-> open calibration futures and freeze or reject the regret bound
-> if and only if calibration passes, stage and seal all target predictions
-> verify the complete target prediction cohort seal
-> open target futures and score without replacement
```

The current execution status is `0/9` calibration objects and `0/12` target
objects opened. Prediction-facing staging and seal generation are implemented;
they may not read any selected future. The runner must execute from a clean,
committed checkout.

## Prediction Construction

One case passes through five outcome-blind boundaries:

1. `prepare_deform360_bias_aware_source.py` performs official camera alignment
   and released robot-pose recovery in a source-data-custodian process. It does
   not create object geometry, tracks, tactile features, or metrics.
2. `stage_deform360_bias_aware_prediction_prefix.py` selects the 81-frame raw
   window from released robot action and openness only. It exports an RGB
   prefix through frame 57, a separate frame-zero episode, and the known
   76-frame robot action. Later object frames are absent from the prediction
   process.
3. `run_deform360_bias_aware_frame_zero.py` reconstructs frame zero only.
   `run_deform360_bias_aware_physical_prior.py` then builds an automatic twin,
   runs frozen driven and zero-action Warp rollouts, and seals the exact
   driven-minus-zero graph-support predictor. Numerical source files, official
   PhysTwin revision, and `real.yaml` are checksum-bound before GPU work. An
   exact persistence fallback is permitted only after a checksummed automatic
   twin returns the declared source-admission failure code; other failures are
   quality failures, not fallback opportunities.
4. `run_deform360_bias_aware_prediction.py` creates the sparse AllTracker
   measurements from causal prefixes, estimates metric covariance using the
   frozen jackknife and cycle rules, selects the raw physical/persistence
   backbone from current observations only, and hashes the source-v4 candidate.
   The state innovation is processed once. It is not reused as prior perception
   reliability.
5. `bpt-deform360-bias-aware-prospective seal-predictions` requires exactly one
   prediction seal or pre-outcome quality-failure seal for every locked case.
   Missing cases, duplicate dispositions, and replacements fail the cohort
   seal.

Dense camera pixels and views are not interpreted as independent samples.
Cycle and leave-one-view disagreement only inflate or invalidate metric
covariance; they cannot establish safety against coherent common-mode camera
bias. That limitation motivates the physical/action support gate and bit-exact
baseline fallback, and remains part of the claim boundary.

Validate or inspect the download plan with:

```bash
bpt-deform360-bias-aware-prospective validate \
  configs/sota/deform360_bias_aware_guarded_belief_prospective_v1.json

bpt-deform360-bias-aware-prospective plan \
  configs/sota/deform360_bias_aware_guarded_belief_prospective_v1.json

# Anonymous snapshot listing may be rate-limited on this large repository.
# This transport lists each already locked object subtree separately while
# preserving the same revision, object allowlist, audio exclusion, and manifest.
bpt-deform360-bias-aware-prospective download-by-object \
  configs/sota/deform360_bias_aware_guarded_belief_prospective_v1.json \
  /path/to/download --manifest /path/to/download_manifest.json

bpt-deform360-bias-aware-prospective seal-predictions \
  configs/sota/deform360_bias_aware_guarded_belief_prospective_v1.json \
  calibration /path/to/predictions \
  /path/to/calibration_prediction_cohort_seal.json
```
