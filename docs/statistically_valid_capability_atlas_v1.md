# Statistically valid decision-capability atlas v1

## Purpose

The exact affine capability atlas is conditional on a registered finite physical support. This module adds an object- or trajectory-level split-conformal correction so that observed real-model discrepancy contracts, but never expands, the model-side atlas.

## Exchangeability unit

Every calibration input must be one scalar maximum for one complete independent physical object or trajectory. All correlated windows, action comparisons, and registered tasks are maximized inside that unit score. Supplying windows as if they were independent calibration samples is outside the contract.

For `n` unit scores and miscoverage `alpha`, the correction is the `ceil((n+1)*(1-alpha))`-th order statistic, clipped below at zero. If that rank exceeds `n`, the API fails closed because the valid deterministic correction is infinite.

## Continuous affine task family

For one proposed action and benchmark, let the real loss difference be affine in the task parameter and let the model-side envelope be a maximum of the witness affine functions retained by the atlas. The maximum real-minus-model difference over a registered task rectangle is a small linear program. The implementation solves it exactly for low-dimensional auditable task families by enumerating all candidate vertices under a strict complexity cap. The optimum may occur at an interior kink of the model envelope, so evaluating only task-box corners is not sufficient.

## Polyhedral composition

If one model-side constraint is

```text
normal @ theta <= offset,
```

and `q_plus` is the nonnegative conformal correction, the calibrated constraint is

```text
normal @ theta <= offset - q_plus.
```

For objective uncertainty `U`, the combined center condition is

```text
normal @ theta_center + support_U(normal) + q_plus <= offset.
```

Thus latent-state ambiguity, objective ambiguity, and observed model discrepancy enter additively while the task capability region remains polyhedral.

## Claim boundary

The finite-sample statement is marginal over a future exchangeable complete physical unit. It requires a target-independent routed score and registered action set, task family, loss, and calibration protocol. It is not conditional coverage for every object category, robustness under arbitrary distribution shift, validation of the physical support or task objective, a safety certificate, or authorization for deployment.
