# Prior-aware guarded observation updates

Bayesian-PhysTwin exposes two complementary identifiability modes for
uncertain-gauge observations.

The existing gauge-aware solver is deliberately strict: it removes every state
direction that can be reproduced by the declared gauge or bias design. This is
a minimax choice and remains the conservative default when nuisance priors are
not independently trustworthy.

`update_prior_aware_gauge_belief` adds a second, opt-in mode. It conditions the
state information on independently calibrated nuisance priors through the Schur
complement

```text
I_state|nuisance = A - B D^{-1} B^T,
```

where `A` is the known-nuisance state information, `B` is the state--nuisance
cross-information, and `D` includes both the nuisance likelihood and nuisance
prior precision. As a gauge prior becomes tight, the result therefore approaches
the known-gauge update continuously. A diffuse prior retains the conservative
fallback behavior.

## Robust group likelihood

The prior-aware update keeps the observation concepts separate:

- association probability remains association support;
- prior reliability is residual-independent feeder evidence;
- prior nominal probability is used inside the nominal/outlier mixture;
- composite weight caps repeated or correlated evidence;
- posterior nominal responsibility is an output of the residual update.

Each declared correlation group receives one nominal/outlier multivariate
Student-t mixture. The degrees of freedom must exceed two because input matrices
are interpreted as covariances. The returned covariance is a working
Laplace/IRLS covariance and still requires prospective coverage evaluation.

## Row-bound physical linearization

`PhysicalLinearizationV1` binds the observation artifact to the physical model
using the complete `(frame, entity, view, window)` row identity. It also records:

- immutable baseline-belief ID;
- action-prefix ID;
- simulator revision;
- observation and future-query Jacobians;
- the action-conditioned physical response.

`build_gauge_aware_batch_from_artifacts` refuses a row permutation or artifact
mismatch. The update-magnitude limit is derived from the physical response stored
in this artifact rather than from an unrelated caller-supplied scalar.

## Nonlinear closure

A local update is not automatically valid for nonlinear PhysTwin propagation.
`evaluate_nonlinear_closure` compares the linearized future query with a nonlinear
replay. The candidate is invalid when both the absolute and relative remainder
exceed their frozen thresholds. The resulting content address can be bound into
the prospective guard decision.

## Complete-belief fallback

`select_complete_belief` routes whole content-addressed beliefs rather than one
NumPy array. The decision binds:

- baseline-belief ID;
- candidate-belief ID;
- common-domain ID;
- source-fitted certificate ID;
- numerical inference admissibility;
- prospective regret acceptance.

A rejected candidate returns the exact baseline object. State, parameters,
particle weights, discrepancy moments, nuisance beliefs, covariance, dtype, and
provenance are therefore not reconstructed from zero correction coefficients.

## Recommended pipeline

```text
source-causal ObservationBeliefV1
        +
PhysicalLinearizationV1
        |
        v
strict-minimax or prior-aware candidate inference
        |
        v
nonlinear PhysTwin closure check
        |
        v
source-fitted baseline-relative regret certificate
        |
        v
complete candidate belief or exact baseline belief
```

Numerical inference admissibility and deployment acceptance remain distinct.
The prior-aware mode does not bypass the prospective guard.
