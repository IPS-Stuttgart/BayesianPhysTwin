# Gauge-aware, query-identifiable observation update

`bayesian_phystwin.gauge_aware_belief` consumes unfused 3-D observation factors
without treating Prob4D window gauges, shared camera bias, or anchor bias as
known. It is a low-dimensional linearized update that decides which part of an
observed innovation is numerically admissible as physical state information
before nonlinear PhysTwin propagation.

## Model

For ordinary observation row `i`,

```text
r_i = H_x,i alpha + H_g,i delta_g + H_s,i beta
      + H_v,i gamma + epsilon_i.
```

For independent-anchor row `a`,

```text
r_a = A_x,a alpha + A_b,a eta + epsilon_a.
```

`alpha` is a physically reachable state coefficient. `delta_g`, `beta`, and
`gamma` are gauge, shared-bias, and view-bias nuisance variables. `eta` is an
optional bias carried by the independent sensor family. Calling a modality
independent of the camera gauge does not imply that its own pixels or time
samples are mutually independent.

## Prior-whitened identifiable subspace

The supplied state prior is factored as `P_x = L_x L_x^T`. Identifiability and
query relevance are evaluated in the standardized coordinates

```text
alpha = L_x z,       z ~ N(0, I).
```

This makes the result invariant to invertible reparameterizations of the state
coefficients. Only the supported, nuisance-distinguishable, query-relevant
subspace of `z` is updated. Every omitted state direction keeps its original
prior mean and covariance; an unidentifiable direction is not reported with
zero uncertainty.

The estimator:

1. keeps conditional point covariance separate from explicit gauge variables;
2. caps information independently within declared observation and anchor
   correlation groups;
3. preserves separate association, prior reliability, nominal-component
   probability, and composite-likelihood fields;
4. permits an explicit anchor-bias design and prior;
5. projects prior-whitened reachable state modes away from prior-supported
   nuisance modes;
6. removes state directions with negligible effect on the requested future
   query;
7. performs a joint Student-t iteratively reweighted Bayesian update;
8. rejects corrections larger than an absolute limit or a declared multiple of
   the action-conditioned physical response;
9. preserves input-artifact lineage in the result;
10. returns zero correction and the unchanged joint prior covariance on
    rejection.

## ObservationBelief adapter

For a versioned `ObservationBeliefV1`, use the repository adapter rather than
manually flattening low-rank factors:

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
standard-normal nuisance coefficient. It does not add the same factor to the
conditional point covariance. When a provider declares
`group_composite_weight_semantics = final-per-row-effective-sample-cap-v1`, that
weight is treated as the final generalized-Bayes row power: Bayesian-PhysTwin
does not apply a second effective-sample cap. Batches from providers without
that declaration retain the consumer-side cap for backward compatibility.
Association probability remains available in
`adapted.association_probability` for reporting and is not substituted for
prior reliability. The adapter records the exact input artifact ID, producer
revision, source digest, case, stream, and exclusive causal frame stop in the
result lineage.

## Anchor dependence

Dense depth pixels, repeated actuator samples, or neighboring tactile elements
must declare correlation groups instead of being treated as independent rows:

```python
batch = GaugeAwareObservationBatch(
    ...,
    anchor_innovation_m=anchor_residual,
    anchor_covariance_m2=anchor_covariance,
    anchor_state_jacobian=anchor_state_jacobian,
    anchor_correlation_group_ids=anchor_groups,
    anchor_prior_reliability=anchor_reliability,
    anchor_composite_weight=anchor_composite_weight,
    anchor_bias_jacobian=anchor_bias_jacobian,
    anchor_bias_prior_covariance=anchor_bias_prior,
)
```

If no anchor groups are supplied, each anchor row receives a distinct group for
backward compatibility. A shared anchor bias can correctly leave a common state
translation unidentifiable.

## Final candidate selection

`result.inference_admissible` means only that the numerical update passed the
identifiability, conditioning, and physical-magnitude checks. It is not an
empirical acceptance certificate.

Final routing additionally requires a source-calibrated baseline-relative
regret decision:

```python
guard = apply_regret_guard(
    baseline,
    candidate,
    target_free_features,
    source_certificate,
)
selection = select_gauge_aware_candidate(
    baseline,
    candidate,
    result,
    regret_decision=guard,
)
```

Omitting the regret decision fails closed to the baseline. A guard decision is
also checked against the exact baseline and candidate arrays before it can be
used. Rejection preserves baseline dtype and bytes.

## Claim boundary

This update is an inference primitive with auditable fallback semantics.
Deployable use still requires an independently frozen regret certificate and a
prospective object- or session-level evaluation. Posterior Gaussian covariance
does not by itself establish calibrated predictive coverage or safety.
