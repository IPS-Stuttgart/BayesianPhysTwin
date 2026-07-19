# Shared nonlinear residual-dynamics source gate

Run date: 2026-07-19

Status: rejected by its frozen source-only gate. No target trajectory, target
metric, or target future artifact was read.

## Registered hypothesis

The candidate is a graph-local MLP trained recursively for twelve residual
steps on complete registered source outcomes. Its inputs combine residual state
and velocity, neighboring state and velocity differences, and the existing
controller/physics features. Three fixed seeds are ensembled. A globally
selected coefficient shrinks the learned rollout toward exact endpoint
persistence, which remains the zero-correction fallback.

Seventeen source interactions are cross-fitted in three whole-case folds. The
gate requires at least 3% balanced improvement over endpoint persistence, both
aggregate metrics to improve, at least two all-case two-metric winning folds,
and no individual metric ratio above 1.05.

## Result

The selected coefficient was `0.25`, meaning that cross-validation already
preferred a prediction close to persistence. It nevertheless failed every
registered transfer criterion:

| Gate quantity | Required | Observed |
| --- | ---: | ---: |
| Balanced improvement | at least +3.000% | -0.402% |
| Aggregate CD change | less than 0% | +0.742% |
| Aggregate track change | less than 0% | +0.063% |
| All-case two-metric winning folds | at least 2/3 | 0/3 |
| Maximum case/metric degradation | at most +5.000% | +5.650% |

The strongest damage occurred on `double_lift_cloth_3` CD (+5.650%). Some
individual cases improved, including both metrics for `single_lift_sloth`,
`single_lift_cloth_1`, `single_lift_cloth_3`, `single_push_rope`, and
`weird_package`, but the signs did not transfer consistently across actions and
objects.

The initial launch used the raw-video extraction root, which does not contain
released `inference.pkl` trajectories, and stopped before loading a source
episode. The recorded run changed only the operational data root to the
checksummed confirmatory-data mirror. The protocol, model, folds, seeds,
candidate blends, and gates were unchanged.

## Interpretation

This rejects the registered graph-local MLP, not residual dynamics in general.
Together with the earlier shared linear null, it indicates that complete
world-frame residual fields are not sufficiently exchangeable across this
heterogeneous 17-case source set for a small pooled predictor to beat endpoint
persistence. Opening the already-examined five targets would add selection bias
without answering that failure, so the target stage remains closed.

A credible successor needs a stronger transferable representation rather than
another tuning pass on this family. The leading options are a material-frame
graph/surface representation with explicit equivariance and contact regimes,
or a published deformable-dynamics backbone integrated behind exact upstream
PhysTwin parity. Prob4D remains useful as a calibrated causal observation model,
but the existing evidence says it should initialize uncertainty over the
current residual, not dictate a persistent future correction.

Machine evidence is archived under
`results/sota/diagnostics/phystwin_shared_nonlinear_residual_source_v1/`.
