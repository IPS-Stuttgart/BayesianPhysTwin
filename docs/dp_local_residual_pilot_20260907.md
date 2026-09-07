# Dirichlet-process local-residual development pilot

## Completed result — 7 September 2026

The workflow is repaired and the real-data comparison completed successfully.
This remains a historical-development experiment, not a promoted method or a
fresh confirmatory paper result.

The selected DP correction scored **7.576843 mm**, versus **7.240300 mm** for
existing ridge: **4.6482% worse**, with one win out of eight validation
trajectories. ExtraTrees scored **6.603346 mm**: **8.7973% better** than ridge,
with eight wins out of eight. The evidence supports investigating the ordinary
nonlinear residual comparator, not promoting this DP implementation.

- Experiment branch: `experiment/dp-local-residual-pilot-20260907`
- Repair/execution commit: `40da25de732387d6f95478a666c307936f8eb7f1`
- Successful workflow: https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34089074496
- Job: `101638727116`; completed `2026-09-07T06:04:30Z`
- Artifact: `10006236763`, `dp-local-residual-pilot-34089074496`
- [Verified summary and all per-trajectory scores](../results/dp_local_residual_pilot_20260907/run_34089074496/summary.json)
- [Workflow](../.github/workflows/dp-local-residual-pilot.yml)
- [Model implementation](../experiments/dp_local_residual_pilot/pilot.py)
- [Development exporter](../experiments/dp_local_residual_pilot/export_development.py)

## What was repaired

The previous run, `34053321725`, failed before rollout because the workflow
passed the nonexistent source directory
`/home/florianpfaff/source-only/deform-bayesian-v1/DEFORM`.
Its training record, source manifest and checkpoint identity checks actually
passed; the initial concern about unmatched metadata was not the final blocker.

The repaired workflow reads the exact upstream revision from the existing v6
protocol, validates the repository and full commit SHA, and uses a separate
checkout at `$GITHUB_WORKSPACE/.dp-upstream/DEFORM`. Sparse checkout materializes
root Python source files only, not `data_set`. The exporter receives that exact
path. It still loads permitted trajectories using the existing manifest paths.

No server data, baseline checkpoint, selection grid, model code, or scientific
protocol was substituted. The upstream revision remains
`b73b8b8ecc033caefa693fab7898741d4e6dbeff`, with a clean tracked checkout.
The existing SHA-256 checks, data-read restrictions and baseline reproduction
gate remain in force. The source checkout is isolated from the preserved server
workspaces.

Offline regression checks on the exact committed workflow blob passed for YAML,
shell and embedded-Python syntax, protocol-driven commit output, rejection of a
branch name instead of a full SHA, rejection of an unexpected repository, and
agreement between checkout and exporter paths. A constructed local Git fixture
verified that the sparse pattern omits datasets while keeping tracked status
clean. These are repair checks, not scientific evidence.

## Verified real-data results

Metric: mean over eight trajectories of the mean absolute coordinate error
across rollout times, all nodes and three coordinates, in millimetres. Lower is
better. Improvement is relative to existing ridge; a negative number is worse.

| Method | Mean L1 (mm) | Improvement vs ridge | Wins vs ridge |
|---|---:|---:|---:|
| Frozen DEFORM hybrid baseline | 7.912030 | -9.2776% | 1/8 |
| Existing ridge correction | 7.240300 | reference | 8 ties |
| Selected finite Gaussian mixture | 7.554476 | -4.3393% | 1/8 |
| Selected finite Bayesian mixture | 7.592074 | -4.8586% | 1/8 |
| Selected DP mixture | 7.576843 | -4.6482% | 1/8 |
| Fixed DP, alpha=1, shrinkage=0.25 | 7.558611 | -4.3964% | 1/8 |
| ExtraTrees residual correction | **6.603346** | **+8.7973%** | **8/8** |

The inner split selected finite K=4 with shrinkage 0.5; finite Bayesian K=8
with concentration 1 and shrinkage 0.5; DP truncation 8 with concentration 10
and shrinkage 0.5; and ExtraTrees with shrinkage 1. The ExtraTrees improvement
relative to the uncorrected hybrid baseline is 16.5404%.

The baseline reproduction gate passed:

- Recorded expected validation L1: `0.007912029745057225 m`.
- Reproduced validation L1: `0.007912029512226582 m`.
- Permitted difference: `1e-7 m`.

The downloaded artifact ZIP's SHA-256 matches GitHub's digest:
`bd815c07ed9774a44a0566ccc1c1262957b667122b2e9c5b55bce987202b07ab`.
The full artifact `results/result.json` has SHA-256
`e708fceaaa897f0439fef45d9d40f3afadb96036c6d0058d357bc760af1c2061`.
All per-trajectory scores, means and win counts were independently recomputed
from saved predictions. Float64 recomputation differs from the original
float32 reductions by at most `8.7408694e-7 mm`, below the `5e-6 mm` roundoff
check tolerance. Prescribed-node predictions are exactly equal to baseline for
all methods, and the exported input NPZ's recorded hash verifies.

## Important limitations

The selected DP's final fits converged for six of eight free-node models;
two reached the configured 150-iteration limit. The fixed DP reference also
converged for six of eight. The finite Bayesian mixture converged for four
of eight, while the selected ordinary finite mixture converged for all eight.
These warnings are retained in the artifact rather than hidden by workflow
success. No post-score iteration-budget or hyperparameter retuning was done.

Every selected DP node model retained all eight components above weight 0.01.
Thus this pilot does not demonstrate adaptive pruning below its truncation cap.
Its input features are reduced to eight whitened principal components, whereas
the ExtraTrees comparator uses the original feature vector. The result concerns
these specific implementations; it is not an isolated proof that the DP prior
causes worse prediction, nor a general rejection of Dirichlet-process models.

The eight validation trajectories were previously used for baseline checkpoint
selection. They are historical development evidence, not fresh confirmation.
No uncertainty calibration or physical-parameter recovery claim follows.

## Implemented comparison and information boundary

The mixture fits a joint distribution of local input features and 3-D residuals
independently for each free material node. Prediction-time component weights
use only the input marginal; Gaussian conditioning supplies the residual mean.
Future free-node residuals never enter the predictor. Prescribed nodes remain
unchanged. The model uses plug-in variational mixture moments, not exact
posterior-predictive integration.

The existing 40 fit trajectories contain 40 distinct causal queries. An inner
32/8 trajectory split selects hyperparameters and shrinkage; selected models
are refitted on the full fit40 before historical validation8 scoring. No causal
query is duplicated across fit40 and validation8. The source-test8 and official
evaluation are excluded by the exporter. No robot, new physical measurements,
active probing, new observation provider or Causal4D integration is involved.

The candidate bank was fixed before scoring: finite mixtures K=1,2,4,8;
finite Bayesian mixtures K=2,4,8; DP truncation 8 with concentration 0.1,1,10;
ExtraTrees; and shrinkages 0,0.25,0.5,1. The current ridge and a fixed DP
alpha=1/shrinkage=0.25 arm provide additional references.

The artifact contains the provenance audit, exported development inputs,
software self-tests, inner scores, frozen selections, final fit diagnostics,
per-trajectory validation predictions and full result JSON. No main-branch
predictor, canonical result or manuscript claim was replaced or promoted.
