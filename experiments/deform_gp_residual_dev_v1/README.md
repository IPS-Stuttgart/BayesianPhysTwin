# Nonlinear GP residual pilot: completed DLO2 development result

## Result and decision

Neither tested Gaussian-process residual corrector improves mean forecasting
error over the existing fixed ridge correction. Retain the incumbent ridge;
this GP construction is not supported as a new accuracy contribution.

| Method | Mean coordinate L1 (mm) | Error increase vs fixed ridge | Wins vs fixed ridge |
| --- | ---: | ---: | ---: |
| Uncorrected DEFORM hybrid | 7.912029 | +9.28% | 1/8 |
| Existing fixed ridge | **7.240300** | Reference | Reference |
| Inner-selected ridge shrinkage | 7.597553 | +4.93% | 3/8 |
| Independent-node GP | 7.581486 | +4.71% | 2/8 |
| Material-aware GP | 7.474714 | +3.24% | 4/8 |

GP-minus-fixed-ridge paired trajectory-bootstrap 95% intervals are
[-0.227362, +0.903876] mm for independent-node GP and
[-0.234720, +0.727294] mm for material-aware GP. Both span benefit and harm:
this small pilot does not establish population-wide inferiority of GPs.

## Matched protocol

The original DLO2/train partition supplies 40 fitting and eight validation
trajectories. A deterministic name-hash split within the 40 fitting trajectories
provides 32 inner-fitting and eight inner-validation trajectories for selecting
GP hyperparameters and the additional ridge-shrinkage control. Selected models
are refitted on all 40 trajectories. The final eight validation outcomes are
not read by the predictor; predictions and selection are hashed before scoring.
That validation cohort already informed historical model development, so this
is retrospective development evidence, not fresh independent confirmation.

The unchanged hybrid physics/learned checkpoint, causal feature builder,
forecast horizons, node roster, and clamped-node values are shared. The fixed
ridge uses penalty 1 and shrinkage 0.25. Its inner-selected control uses the
same penalty and shrinkages {0.125, 0.25, 0.5, 1}.

Both GP variants use a finite-rank Nystrom Matern-3/2 prior. Independent nodes
use 64 inducing variables each; the shared material-aware model uses 512,
matching total latent rank per output coordinate across eight internal nodes.
Every fitting row enters the working likelihood. Inducing selection is seeded,
deterministic, and outcome-blind. Scaling is fitted on the corresponding fitting
partition. The material-aware model multiplies the state kernel by a Matern
kernel on normalized material arc coordinate with fixed length 0.5.

The frozen candidate bank has state lengths {0.5, 1.5}, normalized noise
variances {0.05, 0.5}, and correction shrinkages {0.25, 0.5, 1}. The inner split
selects independent-node length 1.5/noise 0.5/shrinkage 1; material-aware length
0.5/noise 0.05/shrinkage 1; and ridge shrinkage 1.

## Uncertainty boundary

Raw nominal-90% internal-coordinate coverage is 90.7024% for independent-node
GP and 54.0244% for material-aware GP. Independently recomputed mean full interval
widths are 49.0783 and 16.7572 mm, respectively; these widths are posthoc
arithmetic diagnostics. The independent-row working likelihood and finite-rank
prior do not establish calibrated joint or pathwise uncertainty. A GP posterior
mean has a deterministic kernel-ridge equivalent, so point-error improvement
alone would not isolate uniquely Bayesian value.

## Execution and retained evidence

The successful [Actions run 34053472701](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34053472701)
used `gpuserver6000` and source revision
`04da7ccfadb8bc64f3707800a2b10839f29a482e`.
Seven numerical and causal-input tests passed before the comparison.
`import_cache.py` checks the exact permitted fit/validation cache, checkpoint
and manifest identities, trajectory order, array shapes, and historical baseline
reproduction before fitting. It reads no source-test or official-evaluation
arrays. Cached rollouts use the unchanged original evaluation operator.
The CPU GP environment is isolated Python 3.10, NumPy 1.24.3, SciPy 1.15.3.

Independent [verification run 34053801781](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34053801781)
recomputed all five point scores, per-trajectory errors, paired bootstrap
intervals, clamped-node parity, selected configurations, and raw coordinate
coverage from the sealed artifact. It also verified the artifact digest and
prediction/selection hashes. Importer formatting is AST-identical to the
executed source; the GP implementation is unchanged.

Compact records are in [`results/`](results/). The complete artifact
`deform-gp-residual-dev-v1-34053472701-1`, ID `9995263426`, is 17,987,919 bytes
and has SHA-256
`7236cddfbc46dd5ceb977ec6c36296de8614b0392dacf387d14c24931b4873d8`.
It retains input arrays, predictions, fitted GP model arrays, selection records,
source receipts, validation truth, and numerical results.

The checkpoint SHA-256 is
`b64affff638c9d47ca51f17bb7124cc4bd224facd1f7137b0042b7fa9037ea65`;
the source-manifest SHA-256 is
`7c5501997e6bab7b0537ef9cda932ec19312e40618f03a4fad80ffc1622a6d98`.
The paper-side result is recorded in
[BayesianPhysTwin-Paper PR #202](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/pull/202).

No robot data collection, active probing, physical-parameter identification,
production-model replacement, or positive-claim promotion is part of this pilot.
