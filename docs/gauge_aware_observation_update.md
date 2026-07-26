# Gauge-aware, query-identifiable observation update

`bayesian_phystwin.gauge_aware_belief` consumes unfused 3-D observation factors
without treating their Prob4D window gauges as known. It is a low-dimensional
linearized update intended to decide which part of an observed innovation may
enter a physical state belief before nonlinear PhysTwin propagation.

## Model

For factor row `i`, the update uses

```text
r_i = H_x,i alpha + H_g,i delta_g + H_s,i beta + H_v,i gamma + epsilon_i.
```

Here `alpha` is a physically reachable state coefficient, `delta_g` contains
window-gauge nuisance variables, `beta` is shared observation bias, and `gamma`
contains view-specific bias parameters. `epsilon_i` uses the full conditional
3-D covariance. Gauge uncertainty belongs in `gauge_prior_covariance`, not in
that conditional covariance.

The estimator:

1. whitens every factor with its full covariance;
2. caps information within each declared correlation group;
3. appends optional independent anchors that observe state but not camera/gauge
   nuisance variables;
4. projects the reachable state design away from the prior-supported nuisance column space;
5. removes directions that have negligible effect on the requested future
   query;
6. performs a joint Student-t iteratively reweighted Bayesian update;
7. rejects corrections larger than an absolute limit or a declared multiple of
   the action-conditioned physical response;
8. returns an exact baseline copy when the result is rejected.

## Prob4D adapter pattern

The repositories remain loosely coupled. Construct a batch from the neutral
arrays in a Prob4D factor stack:

```python
stacked = factor_bundle.stack()
innovation = stacked.world_mean_m - physical_observation_mean_m

batch = GaugeAwareObservationBatch(
    innovation_m=innovation,
    observation_covariance_m2=stacked.conditional_world_covariance_m2,
    state_jacobian=physical_state_jacobian,
    gauge_jacobian=stacked.gauge_jacobian,
    shared_bias_jacobian=shared_bias_jacobian,
    view_bias_jacobian=view_bias_jacobian,
    query_state_jacobian=future_query_jacobian,
    gauge_prior_covariance=stacked.gauge_prior_covariance,
    correlation_group_ids=stacked.correlation_group_ids,
    prior_reliability=stacked.association_probability,
    physical_response_scale_m=physical_response_scale_m,
)
result = update_gauge_aware_belief(batch)
```

The state, gauge, and bias posterior cross-covariances are retained in
`posterior_covariance`. Causal4D can subsequently apply its finite-support
query-ambiguity gate to the resulting physical particle support. This module
does not replace that nonlinear downstream check.

For a versioned `ObservationBeliefV1`, use the repository adapter rather than
manually flattening the low-rank factors:

```python
adapted = build_gauge_aware_batch_from_observation_belief(
    belief,
    physical_prediction_xyz_m=physical_prediction,
    state_jacobian=physical_state_jacobian,
    query_state_jacobian=future_query_jacobian,
    physical_response_scale_m=physical_response_scale_m,
)
result = update_gauge_aware_belief(adapted.batch)
```

The adapter maps every shared low-rank covariance factor to an explicit
nuisance coefficient with a standard-normal prior. It does not also add that
factor to the conditional covariance. Association probability is available in
`adapted.association_probability` for reporting, but is not used as prior
reliability.

## Claim boundary

The update is an inference primitive, not an empirical acceptance certificate.
A deployable candidate still requires the existing source-fitted regret guard or
a prospectively locked equivalent. Rejected updates can be selected with
`select_gauge_aware_candidate`, which preserves the baseline dtype and bytes.
