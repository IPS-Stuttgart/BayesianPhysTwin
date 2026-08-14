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
checked before it becomes visible at the destination. This makes the artifact
suitable for a future Prob4D-to-BayesianPhysTwin-to-Causal4D conformance corpus
without changing any current provider contract or frozen protocol.

## Diagnostic materialization

`dense_covariance_m2()` exists only for bounded diagnostics and requires an
explicit maximum state dimension. Production query evaluation should use
`project_query_covariance()` so memory scales with the query dimension and
retained ranks rather than quadratically with the number of points.

## Information and claim boundary

The contract and its archive are covariance representations, not uncertainty-
calibration results. A non-null `calibration_artifact_id` records external
calibration lineage; it does not by itself prove coverage. Promotion still
requires the registered object/session-level calibration and held-out physical-
query gates.

This prospective representation does not alter the frozen Deform360 confirmation
protocol. It is intended for a later protocol version or a separately registered
ablation after the current information boundary is complete.
