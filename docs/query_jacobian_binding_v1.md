# Query Jacobian binding v1

`QueryJacobianBindingV1` freezes the exact physical-query linearization passed
to Prob4D. It complements `PhysicalQueryV1`, which defines query semantics and
decision policy, by binding the numerical Jacobian bytes and ordered observation
rows used for a covariance projection.

## Why a separate binding is needed

A projection summary and a generic Jacobian-provider identifier do not prove
which matrix or row ordering generated the summary. Two projections can share
the same query name and dimensions while using different observation rows,
permutations, or Jacobian values.

The binding records:

- query name, component order, physical unit, and coordinate frame;
- source observation and provider-manifest identities;
- exclusive causal frame stop;
- exact Jacobian shape and SHA-256 after canonicalization to contiguous
  little-endian `float64`;
- SHA-256 over the exact ordered row identifiers;
- target-blind and causal-prefix-only declarations; and
- finite JSON metadata.

## Build and freeze

```python
from bayesian_phystwin.query_jacobian_binding_v1 import (
    build_query_jacobian_binding,
    write_query_jacobian_binding,
)

binding = build_query_jacobian_binding(
    query_name="endpoint-displacement",
    component_order=("x", "y", "z"),
    physical_unit="m",
    coordinate_frame="registered-world",
    source_observation_artifact_id=observation.artifact_id,
    provider_manifest_id=provider_manifest_id,
    causal_frame_stop=prefix_stop,
    query_jacobian=query_jacobian,
    row_ids=ordered_factor_point_ids,
)

write_query_jacobian_binding(binding, "query-jacobian-binding.json")
```

Use `binding.artifact_id` as `PhysicalQueryV1.jacobian_provider_id` for new
bound studies. `validate_payload` rechecks actual Jacobian bytes and row order.
Publication is atomic and no-clobber by default.

## Bound Prob4D decision

Prob4D independently validates the portable binding before projecting its
conditional-plus-shared covariance and emits a
`prob4d.bound-query-covariance-projection` receipt. BayesianPhysTwin then uses:

```python
from bayesian_phystwin.bound_query_covariance_decision_v1 import (
    compose_bound_query_covariance_treatment,
)

decision = compose_bound_query_covariance_treatment(
    physical_query,
    binding,
    bound_prob4d_projection_record,
    covariance_value_certificate,
)
```

The composition verifies that query semantics, source observation, provider
manifest, Jacobian digest, row-order digest, covariance descriptors, and
projection dimensions all agree. The resulting existing
`QueryCovarianceTreatmentDecisionV1` stores the bound Prob4D receipt ID as its
`projection_summary_id` and includes the exact lineage IDs in metadata.

The historical `compose_query_covariance_treatment` remains valid for frozen
summary-only evidence. New claim-bearing studies should prefer the bound route.

## Scientific boundary

The binding and receipt establish target-blind numerical lineage only. They do
not establish provider competence, calibrated uncertainty, BayesianPhysTwin
benefit, Causal4D intervention benefit, deployment safety, or state of the art.
The existing source/target separation, proper-score evidence, update guard, and
exact fallback remain mandatory.
