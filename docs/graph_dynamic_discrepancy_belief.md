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

## Source-only tournament boundary

`bayesian_phystwin.graph_dynamic_discrepancy_tournament` connects a graph
forecast to the candidate-agnostic source-only tournament without importing
scored outcomes into candidate generation.

The information order is explicit:

```python
from bayesian_phystwin.graph_dynamic_discrepancy_tournament import (
    GraphDynamicTournamentScoringPolicyV1,
    build_graph_dynamic_tournament_prediction_bundle,
    score_graph_dynamic_tournament_prediction_bundle,
    seal_graph_dynamic_tournament_prediction,
)

prediction = seal_graph_dynamic_tournament_prediction(
    forecast,
    selected_horizon_index=0,
    candidate_id="graph_modal",
    unit_id="object-a-endpoint",
    group_id="object-a",
    horizon="endpoint",
    source_revision=source_revision,
    configuration_sha256=configuration_sha256,
    prediction_barrier_sha256=prediction_barrier_sha256,
    physical_fallback_mean_m=fallback_mean,
    physical_fallback_covariance_m2=fallback_covariance,
    graph_rank=graph_rank,
    parameter_count=registered_parameter_count,
    runtime_milliseconds=runtime_milliseconds,
    accepted=guard_accepted,
    reason=guard_reason,
)
bundle = build_graph_dynamic_tournament_prediction_bundle(predictions)

# Scored targets enter only after the complete prediction roster is sealed.
scored = score_graph_dynamic_tournament_prediction_bundle(
    bundle,
    targets_by_registered_unit,
    scoring_policy=GraphDynamicTournamentScoringPolicyV1(),
)
```

The seal constructor has no target argument. Each prediction retains the
complete queried multi-horizon mean and covariance, the selected horizon, the
exact physical fallback, the source revision, the frozen configuration, and the
common prediction barrier. Arrays are copied into bytes-backed immutable
storage. A bundle requires a complete unique unit roster and gives the candidate
one content identity across all source groups; the tournament candidate spec
uses that bundle identity rather than a different artifact ID per scored unit.

Post-outcome scoring emits frozen `TournamentRecord` values for both the graph
candidate and the physical fallback. A rejected graph prediction deploys the
same fallback point loss, proper score, interval coverage, and interval width as
the common fallback record. This allows the generic tournament parser to verify
exact fallback and candidate-independent fallback values.

The default registered score policy uses:

- endpoint coordinate RMSE;
- full joint Gaussian negative log likelihood per coordinate, with one explicit
  positive eigenvalue floor; and
- coordinatewise marginal 90% intervals.

The policy is itself content addressed. Intervals can be disabled only by
disabling both the nominal coverage and its standard score, in which case the
generic tournament must register interval semantics as `none`.

This adapter establishes the pre-outcome seal and common record shape. It does
not supply source outcomes, manufacture records for the dynamic or structured
candidates, or decide promotion. The complete source-only runner must combine
the sealed graph bundle with the physical fallback, last residual,
horizon-conditioned dynamic candidate, and structured candidate under one
registered unit roster and one scoring policy.

## Scientific boundary

This implementation adds a reusable dynamic discrepancy belief, controlled
nested-baseline tests, and a fail-closed source-tournament adapter. It does not
change any frozen PhysTwin result, establish raw covariance calibration,
demonstrate fresh-object transfer, establish Causal4D benefit, or support a
deployment or state-of-the-art claim.
