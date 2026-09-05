# Passive occlusion covariance v1

## Question

Does a meaningful joint predictive-error belief improve hidden-node geometry when the visible subset is fixed in advance? No robot, new acquisition, active probe, or adaptive observation choice is used.

The experiment reuses immutable DEFORM DLO4/DLO5 hybrid forecasts from run `33361441865`. The parent is the trained DEFORM hybrid plus the existing local residual, not bare physics. Eight previously held source trajectories per DLO fit a new 96-dimensional predictive-error second moment, compressed to rank eight plus a diagonal component. All 14 already-open evaluation trajectories per DLO are scored, without target fitting or case selection. This is retrospective, not new independent confirmation.

Each of 40 predetermined anchors reveals only the current visible free-node positions. Central masks hide 25%, 50%, or 75% of the eight free nodes. Outputs are reconstructed current geometry and forecasts 100, 250, and 500 ms later. Every anchor starts from the same unchanged prior; this isolates conditioning and does not feed corrected states back into the simulator.

## Matched controls

All covariance arms share exactly the same prior means and coordinate variances. The diagonal control removes cross-node/time dependence. Four fixed sign-scrambled covariance controls preserve positive definiteness, eigenvalues, and variances; their losses, not predictions, are averaged. Two deterministic controls interpolate or globally propagate visible residuals and receive source-fitted gains separately for every mask and horizon.

The primary condition is 50% hidden, current-time reconstruction. The primary score is complete-trajectory hidden-node Euclidean 3D RMSE. A positive primary result must beat diagonal, mean scrambled loss, and source-tuned interpolation with paired 98.333% bootstrap intervals below zero (three-comparison Bonferroni familywise alpha 0.05). All other conditions and the global baseline are retained regardless of sign. Whole trajectories, not points or frames, are resampled within DLO4/DLO5.

## Execution and evidence

The workflow `.github/workflows/passive-occlusion-covariance-v1.yml` responds only to push changes in `.github/requests/passive-occlusion-covariance-v1.json` on the experiment branch. It uses `[self-hosted, gpuserver4090]`, read-only repository permissions, immutable action revisions, and an isolated NumPy environment. It does not mutate datasets or the parent experiment.

The evaluator verifies eight parent file identities and every trajectory hash. Both source models are sealed before target readout use. All six DLO/mask prediction archives are sealed before hidden-value scoring. The native pickle IO adapter necessarily deserializes a whole record, but the inference function accepts only a prior and current visible coordinates. Synthetic tests replace all hidden and future truth and verify bit-identical inference.

Artifacts contain source models, predictions, seals, per-trajectory CSV/JSON, full summaries, and source/run identities. Technical failures and negative scientific results are distinct. Scientific negativity does not fail the software job.

## Boundaries

The new covariance is a source-fitted empirical predictive-error model, not an exact posterior over material parameters. The source residuals come from the prior 39-fit hybrid; evaluation means come from the parent's all-56 retraining. Neither hybrid is retrained here. The assumed 1 mm observation standard deviation is a likelihood regularizer, not measured calibration. The observations are recorded marker coordinates, not a tested camera-occlusion perception system.

A deterministic optimizer using the identical full Gaussian model reproduces its conditional mean; the self-test verifies this. The experiment can isolate the value of retaining meaningful joint dependence, but cannot prove superiority over every deterministic representation. Two DLO objects do not establish generalization to an arbitrary object population.

## Local invariant tests

```bash
python scripts/remote/run_passive_occlusion_covariance_v1.py --self-test
```

The actual real-data request binds the exact protocol and evaluator SHA-256 values. Do not edit the scientific recipe or retune on the evaluation results and present a replacement as the same test.
