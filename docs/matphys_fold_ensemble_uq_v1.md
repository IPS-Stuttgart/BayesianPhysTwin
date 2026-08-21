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

Official Warp is empirically nondeterministic, so one replay per fold is not a
valid estimator of checkpoint disagreement. The registered v2 replay therefore
uses at least four executions per spring field and applies the law of total
variance:

```text
Sigma_total = Cov_j(E[X | fold j]) + E_j(Cov[X | fold j]).
```

The first term is between-checkpoint disagreement; the second is the measured
within-checkpoint Warp replay floor. Duplicate source checkpoint files remain
prohibited. The raw total covariance is epistemic and numerical-backend
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

The initial one-replay parity gate failed as intended: two incumbent executions
differed by 0.243 mm coordinate RMSE (3.52 mm maximum), while member spread was
1.38 mm RMS, a 5.7x signal-to-replay ratio. This falsifies byte-level Warp
determinism but leaves enough source signal to test the repeated-replay model.
The failed artifact remains immutable; v2 does not reinterpret it as a pass.

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

The registered feature adapter is
`all-calibrated-frame-zero-rgb-mask-depth-dinov2-graph-parts-v1`. It projects
the immutable physical graph into every calibrated camera, admits a node-view
only when its frame-zero projection lies in the object mask and agrees with
rendered metric depth within 20 mm, samples the pinned DINOv2-L/14 patch field,
and fills unseen nodes from the nearest directly observed graph node. The
existing deterministic semantic-geodesic partition then produces five
connected parts. Deform360's metadata-only `sheet` stratum maps to MatPhys's
cloth class; `volumetric` remains uniformly uncertain over the three public
volumetric training classes. Every RGB, mask, depth, calibration, graph, model,
and output digest is recorded. No frame after frame zero contributes to this
part artifact.

On the opened `153-cake` source smoke, the fixed visibility rule directly
supports 655 of 762 graph nodes across 32 cameras before any DINO inference.
This is an interface-support diagnostic only; no future error, calibration, or
target claim follows from it.

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
