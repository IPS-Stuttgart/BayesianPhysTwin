# Guard-aware uncertainty with an unchanged point forecast

## Scope and hypothesis

This is a new, exploratory development experiment on the already-open DLO1,
DLO2, and DLO3 trajectories. It follows, and does not revise, the failed frozen
forecast-sensing and weak-constraint belief studies. No protected data, DLO4/5,
official DLO3 evaluation, physical recordings, or new target is authorized.
All existing successful point forecasts remain immutable. Outcomes remain local
or private-paper evidence only. No arm can authorize automatic promotion.

The weak-constraint experiment found that multiplying the tangent covariance by
the squared mean-guard gain left almost all reported uncertainty in a fixed
isotropic floor. A cap on the deployed mean is not evidence that uncertainty
about the physical system has vanished. This motivates a **separate** test; it
does not retrospectively repair or relabel the previous failed experiment.

## Fixed-mean Gaussian risk surrogate

Let the shadow local physical belief have mean `mu` and covariance `C`, while the
unchanged deployed point forecast is `a`. Its second moment about that forecast is

```text
S(a) = E[(Y-a)(Y-a)^T | prefix] = C + (mu-a)(mu-a)^T.
```

For a Gaussian report constrained to have mean `a`, expected negative log
likelihood, up to a constant, is

```text
0.5 * [log det(Sigma) + tr(Sigma^-1 S(a))].
```

For positive definite `S(a)` this is minimized at `Sigma = S(a)`: whitening by
`S(a)` reduces the difference to a sum of `log(lambda) + 1/lambda - 1 >= 0`.
The second-moment identity does not require a Gaussian shadow distribution.
Neither identity makes a misspecified shadow belief correct. This is a
fixed-mean Gaussian log-score surrogate, **not** the exact posterior covariance,
not a distribution-free coverage result, and not a new mathematical identity.
The contribution being tested is whether this reporting contract adds useful,
transferable uncertainty to an otherwise unchanged physics-based predictor.

The shadow belief is the already-sealed **strong_8** posterior: 12 standardized
position and 12 velocity coefficients, jointly inferred with a three-dimensional
shared observation bias from four nodes at each of prefix frames 41 and 49.
The bias is marginalized, not physically propagated. Its full physical 24x24
posterior block retains correlations with the nuisance through conditioning.
Using the already-sealed native finite-difference response `J`:

```text
mu = incumbent + J * physical_posterior_mean
V  = J * physical_posterior_covariance * J^T
C  = V + (0.003 m)^2 I
a  = previous_paired_8                   # byte-identical frozen array
```

`J` is a local linearization at the registered injection magnitude, not an exact
description of nonlinear contact/rod dynamics. An unguarded shadow mean may be
poor. There are no additional queries, native runs, parameter fits, mean updates,
new sensors, or changes to learned readouts. All arms share a single mean array;
its registered dtype, shape, C-order bytes, and SHA-256 must match the parent.

## Frozen comparisons

Five raw covariance carriers are sealed for every trajectory before new scoring:

1. `isotropic`: `(0.003 m)^2 I`.
2. `guard_scaled`: `gain_strong8^2 V + (0.003 m)^2 I`.
3. `shadow`: `C`, without unresolved mean discrepancy.
4. `fixed_mean_bridge` (primary): `C + (mu-a)(mu-a)^T`.
5. `rotated_bridge`: the primary covariance with a fixed cyclic XYZ permutation.
   This preserves raw eigenvalues and volume and is a direction-value control.

A sixth comparator, `source_full`, is fitted **only** on the 13 non-design DLO2
trajectories: equal-trajectory uncentered error second moment within each horizon,
plus a fixed `(0.000001 m)^2 I` numerical ridge. It does not subtract the error
mean, because the deployed mean is fixed. This is a deliberately strong, cheap
full-matrix comparator, not an isotropic-only uncertainty leaderboard.

All six receive the same two declared calibration procedures, separately for
early/middle/late 40-frame horizons. Primary scalar calibration is source mean
NEES / 3. Secondary conformal scaling uses each source trajectory's 90th
percentile NEES with `higher`, then rank 13/13 (the maximum), divided by
`chi2_3(.9)=6.251388631170325`. No calibration choice is selected on transfer data.
Scalar calibration can shrink the raw matrices; no post-calibration global
conservativeness or Loewner-monotonicity guarantee is asserted.

## Denominator, estimands, and advancement gate

The exact inherited roster is 30 trajectories: 14 DLO2 including the excluded
design case, eight DLO1, and eight DLO3. Prediction accounting retains all 30.
Analysis uses 13 source calibration trajectories and 16 opened transfer
trajectories. Hidden identities are 3/5/7/9, frames 50:170 (raw 52:172), metres
and metres squared in the frozen world frame. The known clamp trajectory and two
initial full states are shared parent inputs. The object-specific physical and
readout checkpoints are already fitted; this is not zero-shot parameter transfer.

Report equal-event-within-trajectory marginal 3D Gaussian NLL, NEES, 90% ellipsoid
coverage/volume, geometric-mean full ellipsoid-axis diameter, and horizon results.
Average trajectories equally within objects and objects equally in aggregates.
Do not treat coordinates, identities, or frames as independent replicates. Use
10,000 paired whole-trajectory bootstrap resamples, seed 260835. With only two
transfer objects these intervals do not identify population-level object transfer.

The primary `fixed_mean_bridge__moment` must pass **every** check on **each**
DLO1 and DLO3 separately:

- NLL-difference 95% upper bound strictly below zero versus isotropic, source-full,
  and unguarded shadow covariance, all with exactly the same point mean.
- At least five/eight lower-NLL trajectories versus isotropic and source-full.
- Coverage within [0.80, 0.98].
- Mean ellipsoid volume no larger than isotropic and source-full.
- Exact point-array identity and complete failure accounting.

Secondary conformal/rotation/scaled-covariance results cannot rescue a failed
primary. No empirical failure is replaced. Missing or changed carriers stop the
run without modifying the old forecast. A failure retains the previous paired
point predictor and the original uncertainty controls; it is not a license to
search calibration variants on this cohort.

## Verification before and after execution

Before data-dependent computation, freeze this protocol, implementation, tests,
and independent verifier at a clean local Git commit. Synthetic tests cover the
second-moment identity, expected fixed-mean Gaussian score optimum, zero-guard
uncertainty retention, correlated physical/nuisance covariance, rotational
equivariance, exact mean bytes, PSD/finite checks, source-only calibration, and
failure-closed gates. This is implementation evidence, not an empirical result.

The protocol binds one fresh output root and a separate write-once attempt
ledger consumed before prediction input access. Alternate roots and a second
prediction invocation are rejected. Validation and independent recomputation do
not create a new empirical prediction or change the sealed outputs.

Prediction consumes only hash-bound parent model/fit/prediction carriers, not
truth. Require all object seals before opening DLO2 scoring truth. Seal source
calibration before opening already-authorized DLO1/DLO3 scoring truth. Independently
recompute the covariance construction, source calibration, marginal scores,
bootstrap intervals, gates, and mean hashes. Preserve the complete parent evidence.
