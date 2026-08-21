# MatPhys fold-ensemble backend uncertainty v1

## Purpose

This is an opt-in MatPhys qualification track. It does not replace the current
DEFORM point predictor or alter any frozen Causal4D claim. Its paper question is
narrower:

> Can disagreement among target-excluded, source-trained physical models add
> useful predictive covariance to a strong unchanged mean forecast?

The public MatPhys checkpoint is not the candidate. It saturated all springs at
its lower 1000 Pa limit on the released PhysTwin control and remains a frozen
negative result. The candidate is the 11-fold source-supervised family trained
with one PhysTwin object held out per fold. Each fold predicts a bounded
log-stiffness residual around the exact incumbent spring field, and official
PhysTwin/Warp propagates that field into a trajectory.

## Method boundary

For fold `j` and incumbent spring field `k_0`,

```text
k_j = k_0 * exp(alpha * log(2) * tanh(r_j)),  alpha in [0, 1].
```

Thus every fold stays within a factor of two of the incumbent. `alpha = 0`
returns the same incumbent array object without arithmetic. The target object
must be absent from every fold's training-object list, and every checkpoint and
training audit is SHA-256 bound.

The fold trajectory ensemble supplies population moments after exact duplicate
members are collapsed. Duplicate files or trajectories cannot masquerade as
additional independent evidence. The raw ensemble covariance is epistemic
evidence, not a calibrated posterior.

For the confirmatory UQ comparison, the preferred candidate keeps the selected
DEFORM mean byte-identical and uses the calibrated MatPhys/Warp ensemble spread
only as a covariance donor. This isolates the Bayesian contribution and ensures
that adding MatPhys cannot degrade point accuracy. A separate MatPhys-mean arm
is diagnostic only unless it passes the source transfer gate.

## Evidence stages

### A. Opened-source interface smoke

Run all 11 folds on one already-open Deform360 source interaction. Verify:

- exact checkpoint and training-audit hashes;
- causal 16-frame sampling ending before the registered prefix boundary;
- canonical graph edge order and 11-D MatPhys geometry features;
- finite bounded spring fields from all members;
- nonzero between-fold spread;
- exact zero-strength incumbent identity;
- finite official Warp replays.

The first smoke may use the explicitly labeled
`geometry-voronoi-zero-part-feature-control-v1`. That control proves runtime and
graph competence only. It cannot authorize a transfer or paper claim.

### B. Opened-source scientific gate

Before fresh evaluation, use registered causal 1024-D part features and a
custom disjoint-camera outcome reconstruction on opened source objects. Freeze
the covariance map and any scalar calibration using source cases only. Require:

- all checkpoint members remain target-excluded;
- the MatPhys covariance is finite and positive semidefinite;
- no point-mean change in the covariance-only arm;
- lower source NLL than the unchanged-mean isotropic and split-conformal
  comparators;
- 90% coverage within the preregistered acceptance interval without wider mean
  intervals than the comparator;
- no source case is silently replaced; and
- failed providers use the exact unchanged baseline.

### C. Fresh custom Deform360 evaluation

Only after stage B passes, select genuinely fresh physical objects using names
and hash-only exclusion manifests. Seal predictions before disjoint scoring
reconstructions or outcomes are opened. Because fresh raw Deform360 objects do
not have official processed annotations, this is a custom preregistered
calibration/forecast study, not an official Deform360 SOTA claim.

Primary comparisons:

1. unchanged DEFORM mean plus registered isotropic covariance;
2. unchanged DEFORM mean plus source-only split conformal calibration;
3. unchanged DEFORM mean plus calibrated MatPhys/Warp ensemble covariance;
4. diagnostic MatPhys/Warp ensemble mean and covariance.

Report object-session-balanced NLL, marginal 90% coverage, interval width,
energy score, NEES, and a predeclared risk-sensitive decision metric. Preserve
ordinary successes, exact-fallback technical failures, and unsealable cases as
separate counts over the complete locked denominator.

## Paper contribution if successful

The novelty is not “MatPhys predicts better springs.” It is a guarded,
backend-independent Bayesian construction that turns source-trained physical
model disagreement into useful uncertainty around a stronger unchanged mean,
with exact fallback and prospective calibration evidence. DEFORM supplies the
best current mean; MatPhys supplies a mechanistically distinct epistemic signal.
That combination addresses an unoccupied part of the deformable-dynamics
literature without weakening the existing DEFORM result.

If the source gate fails, retain the result as evidence that the current folds
mostly encode global softening and do not transfer as a calibrated covariance
donor. Do not open a fresh target cohort in that case.
