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

## Diagnostic materialization

`dense_covariance_m2()` exists only for bounded diagnostics and requires an
explicit maximum state dimension. Production query evaluation should use
`project_query_covariance()` so memory scales with the query dimension and
retained ranks rather than quadratically with the number of points.

## Information and claim boundary

The contract is a covariance representation, not an uncertainty-calibration
result. A non-null `calibration_artifact_id` records external calibration
lineage; it does not by itself prove coverage. Promotion still requires the
registered object/session-level calibration and held-out physical-query gates.

This prospective representation does not alter the frozen Deform360 confirmation
protocol. It is intended for a later protocol version or a separately registered
ablation after the current information boundary is complete.
