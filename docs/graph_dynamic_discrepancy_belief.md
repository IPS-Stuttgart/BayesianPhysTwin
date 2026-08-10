# Graph-modal dynamic discrepancy belief

`bayesian_phystwin.graph_dynamic_discrepancy` is an experimental, NumPy-only
belief over predictive readout/model discrepancy. It does not reinterpret a
visual residual as a latent physical-state correction.

## Model

For an orthonormal graph basis `Phi`, the node discrepancy is

```text
d_t = Phi c_t
```

and the modal dynamics are

```text
c_(t+1) = c_t + dt v_t + 0.5 dt^2 a_t + w_position
a_(t)   = an optional predeclared causal modal input
v_(t+1) = retention v_t + dt a_t + w_velocity.
```

The process covariance combines an independent position innovation with an
integrated modal-acceleration innovation. The complete modal position/velocity
covariance is retained, so registered node queries preserve cross-node and
cross-horizon correlations.

## Robust recursive update

`fit_graph_dynamic_discrepancy` consumes only the supplied residual prefix. Each
frame is predicted before its observation update. Rows retain separate
residual-independent reliability values. Each declared correlation group
receives one covariance-parameterized, downweight-only Student-t factor. The
effective-sample cap enters once as a generalized-Bayes group power and does
not alter the residual used to determine whether that group is an outlier.

The update works in the stochastic subspace of the predicted positive-
semidefinite prior. Deterministic modal directions, including a velocity mode
that has decayed exactly to zero, therefore remain deterministic instead of
being revived by numerical jitter.

The update is admitted only when the reweighted fixed point is stationary,
the posterior system is well conditioned, and the inferred node position and
velocity fields remain within the configured plausibility limits. Rejection
returns the exact predicted mean and covariance for that frame.

The returned covariance is a working Gauss-Newton/IRLS covariance. It is not the
exact Student-t posterior Hessian and requires independent prospective coverage
and proper-score evaluation before claim-bearing use.

## Nested baselines

Two constructors make matched deterministic and Bayesian baselines exact
special cases:

- `from_last_residual` produces a deterministic held-residual forecast;
- `from_independent_endpoint_posterior` reproduces the current independent
  random-walk endpoint mean and variance when velocity and acceleration are
  disabled.

These identities make it possible to compare richer graph dynamics without
changing evaluation, fallback, or baseline semantics. Their identity-basis
representations are parity tools and scale quadratically with the number of
tracks; production use should pass a compact graph basis.

## Registered forecast queries

`belief.forecast(...)` accepts strictly increasing positive horizon steps and an
optional subset of graph nodes. It returns the complete joint Gaussian
covariance in horizon-major, node-major, coordinate-major order. Dense query
materialization is guarded by an explicit byte budget.

An optional deterministic `modal_acceleration_mps2` input can encode a
predeclared causal action response. The module does not fit such an action model
and does not authorize target-informed action selection.

## Scientific boundary

This implementation adds a reusable dynamic discrepancy belief and controlled
nested-baseline tests. It does not change any frozen PhysTwin result, establish
raw covariance calibration, demonstrate fresh-object transfer, establish
Causal4D benefit, or support a deployment or state-of-the-art claim.
