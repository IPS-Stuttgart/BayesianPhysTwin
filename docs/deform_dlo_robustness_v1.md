# DEFORM DLO robustness v1

This protocol transfers the already fixed DLO2 recipe to DLO3, which has not
been used for BayesianPhysTwin development. DLO4 and DLO5 are unopened reserves
and cannot replace DLO3. The upstream repository contains its authors' own
training logs for some DLOs; those files are not project development evidence
and are not read by this study.

Before any DLO3 trajectory payload is read, the protocol fixes the physical
training budget, seed-42 primary candidate, seeds 43 and 44 stability audit,
ridge `1.0`, shrinkage `0.25`, causal feature contract, exact physical fallback,
and all reporting operators. Prob4D is unused. A domain-separated hash of each
training basename assigns 39 fit, 9 calibration, and 8 source-test trajectories.
The source-test payload is opened only after predictions are sealed.

The source gate requires at least 1% improvement over the identically trained
physical checkpoint, at least 6/8 wins, worst-case ratio at most 1.10, and mean
L1 below 7.7 mm. At least two of three fixed seeds must pass the registered
stability criteria. Only then may an all-56 refit and one-shot DLO3 evaluation
be prepared. DLO3 evaluation cannot select a seed, feature set, covariance,
solver setting, backend, case, or retry.

The mechanism audit separates the physical backbone, action conditioning,
local coordinate frame, intercept-only correction, shrinkage, and persistence
backbone. The compute control spends the measured residual-fit wall time on
additional official DEFORM updates using the same schedule continuation. The
solver audit evaluates 5/10/20 PBD iterations and joint bend/twist multipliers
of 0.9/1.0/1.1 without selecting among them.

Backend portability is a separately gated PyElastica 1.0.0 Cosserat-rod arm.
Its finite parameter bank is selected on fit trajectories only; DLO3 target
access is allowed for that arm only if its calibration and source-test gate
passes. Failure leaves the official DEFORM candidate unchanged.

The Bayesian audit leaves the point mean unchanged and adds full 3x3
coordinate covariance from trajectory-clustered coefficient and residual
uncertainty. Nine calibration trajectories set the 90% scale by the maximum
trajectory score. Diagonal, coefficient-only, residual-only, pooled-isotropic,
uncalibrated full-covariance, and calibrated full-covariance distributions are
all compared without target selection. Temporal independence is not claimed.

Outcome-bearing compact artifacts belong only in the private
`BayesianPhysTwin-Paper` repository. The public repository retains executable
code, the frozen protocol, and target-blind provenance receipts.

Protocol: `configs/sota/deform_dlo_robustness_v1.json`.
