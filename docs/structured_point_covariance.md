# Structured point covariance

`StructuredPointCovarianceV1` represents uncertainty over identified metric 3-D
points without pretending that coherent object-level uncertainty is independent
from point to point.

## Representation

For points \(p_1,\ldots,p_N\), the covariance is

\[
\Sigma = \operatorname{blockdiag}(D_1,\ldots,D_N)
         + \sum_c U_c U_c^\mathsf{T}.
\]

- `local_covariance_m2[n]` is the conditional local \(3\times3\) block \(D_n\).
- `shared_factors_m[c]` is an \(N\times3\times r_c\) root for one labeled
  shared component.
- Supported labels are `discrepancy`, `camera_bias`, `gauge`, `between_model`,
  and `process`.

The local blocks **exclude** all retained shared components. Adding a shared
contribution into the local blocks as well as its low-rank root would double
count uncertainty and violates the contract.

Different component labels are treated as additive independent roots. Correlated
modes must be represented together in one root rather than split across labels.
The labels preserve provenance and decomposition; they do not assert that the
corresponding mechanism has been physically identified.

## Query propagation

For a linearized physical query with Jacobian \(J\), the implementation computes

\[
\Sigma_q = J D J^\mathsf{T}
         + \sum_c (J U_c)(J U_c)^\mathsf{T}
\]

without constructing the full \(3N\times3N\) covariance. The returned object
retains:

- the local contribution;
- every projected shared factor;
- every component covariance; and
- their total.

This supports physical-query uncertainty audits, component ablations, and later
object/session-level calibration while retaining coherent cross-point modes.

## Exact `ObservationBeliefV1` bridge

`ObservationBeliefV1` uses a different but compatible dependence convention:
rows with one `factor_group_id` share the declared latent factor columns, while
different factor groups are independent. The
`observation_structured_covariance` adapter expands each factor group into a
disjoint column block and then groups factor columns under an explicit
caller-supplied component classification:

```python
from bayesian_phystwin.observation_structured_covariance import (
    structured_covariance_from_observation_belief,
)

covariance = structured_covariance_from_observation_belief(
    observation,
    coordinate_frame="world",
    factor_components={
        name: "gauge" for name in observation.factor_names
    },
)
```

For factor groups \(g\) and component-specific column selections \(I_c\), the
adapter constructs

\[
U_c = [U_{c,1},\ldots,U_{c,G}],
\]

where \(U_{c,g}\) contains the original columns \(I_c\) on rows belonging to
factor group \(g\) and zeros elsewhere. Consequently,

\[
\operatorname{blockdiag}(D_n) + \sum_c U_cU_c^\mathsf{T}
\]

is exactly the covariance represented by the observation artifact. The adapter
performs no truncation, rescaling, calibration, or inference. It fails closed
when:

- a factor name is unclassified or classified more than once;
- a component label is outside the frozen roster;
- factor names are ambiguous duplicates; or
- the exact group expansion exceeds the caller's rank budget.

The coordinate frame is mandatory rather than inferred from metadata. The
result binds the source observation artifact identity, provider revision,
causal frame stop, component mapping, group count, expanded ranks, and caller
metadata.

## Portable archive boundary

`structured_point_covariance_io` preserves the complete decomposition in a
strict NPZ archive instead of reducing it to dense covariance or independent
marginals:

```python
from bayesian_phystwin.structured_point_covariance_io import (
    load_structured_point_covariance,
    write_structured_point_covariance,
)

write_structured_point_covariance("covariance.npz", covariance)
reloaded = load_structured_point_covariance("covariance.npz")
assert reloaded.artifact_id == covariance.artifact_id
```

The archive contains one exact `float64` local-block array, one exact `float64`
root per declared shared component, and a strict finite JSON descriptor that
binds the original `StructuredPointCovarianceV1` content identity. The loader:

- rejects duplicate JSON keys, non-finite constants, unknown fields, unsupported
  component labels, missing or extra archive members, and dtype or shape drift;
- enforces archive, decoded-byte, point-count, shared-rank, descriptor, and
  compression-ratio budgets before accepting the artifact;
- rejects symbolic links and files that change while being read; and
- reconstructs the covariance through its normal validator and verifies the
  exact original descriptor and artifact identity.

Publication is atomic and no-clobber by default. Deliberate replacement requires
`overwrite=True`, and a completed temporary archive is loaded and identity-
checked before it becomes visible at the destination. The convenience function
`write_observation_structured_covariance()` performs the exact observation
bridge and verified publication as one operation.

Together, the adapter and archive make the named uncertainty budget suitable for
a future Prob4D-to-BayesianPhysTwin-to-Causal4D conformance corpus without
changing any current provider contract or frozen protocol.

## Diagnostic materialization

`dense_covariance_m2()` exists only for bounded diagnostics and requires an
explicit maximum state dimension. Production query evaluation should use
`project_query_covariance()` so memory scales with the query dimension and
retained ranks rather than quadratically with the number of points.

## Information and claim boundary

The contract, adapter, and archive are covariance representations, not
uncertainty-calibration results. A non-null `calibration_artifact_id` records
external calibration lineage; it does not by itself prove coverage. Promotion
still requires the registered object/session-level calibration and held-out
physical-query gates.

This prospective representation does not alter the frozen Deform360 confirmation
protocol. It is intended for a later protocol version or a separately registered
ablation after the current information boundary is complete.
