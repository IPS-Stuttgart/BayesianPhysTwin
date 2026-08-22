# Query estimability certificate v1

## Purpose

`bayesian_phystwin.query_estimability_certificate_v1` separates two questions
that are often conflated:

1. Is the registered query identifiable after declared nuisance directions are
   removed?
2. If it is identifiable, can it be recovered without excessive amplification
   of observation noise?

The existing
[`QueryIdentifiabilityCertificateV2`](query_identifiability_certificate_v2.md)
answers the first question. This additive certificate answers the second while
binding the exact identifiability artifact rather than accepting an independent
copy of its physical design, nuisance design, query map, or factor operator.

The module is an experimental direct import. It does not change the frozen v2
identifiability semantics, add package-root symbols, or alter any target or
confirmation protocol.

## Local noise map

The source certificate works in whitened observation coordinates and records

\[
A=(I-P_N)X,
\qquad
M=BA^\dagger,
\]

where \(P_N\) projects onto the declared nuisance span. If the original
whitened observation noise has unit covariance, the effective map from that
noise to the registered query is

\[
G=M(I-P_N).
\]

The induced local query covariance is therefore

\[
C_{q,\epsilon}=GG^\top.
\]

In exact arithmetic, the rows of \(M\) already lie in the nuisance-orthogonal
observation subspace, so \(G=M\). The implementation nevertheless applies the
projector explicitly and reports the Frobenius norm of \(MP_N\) as a numerical
nuisance-leakage diagnostic.

This covariance is not a calibrated posterior covariance. It is the query
variance induced by one unit of noise in the exact whitened local observation
model bound by the source certificate.

## Query scale and dimensionless gain

A query can mix coordinates with different units or tolerances. The caller must
therefore provide an exactly symmetric positive-definite query scale
\(S_q\), together with a content identity for that scale. The matrix may encode,
for example, squared engineering tolerances or a source-frozen reference
covariance.

Let

\[
S_q=LL^\top
\]

be its admitted Cholesky factorization. The normalized noise operator is

\[
\bar G=L^{-1}G.
\]

The certificate reports

\[
g_{\max}=\lVert\bar G\rVert_2
\]

and

\[
g_{\mathrm{rms}}
=
\frac{\lVert\bar G\rVert_F}{\sqrt{d_q}},
\]

where \(d_q\) is the query dimension. Equivalently,
\(g_{\max}^2\) is the largest generalized eigenvalue of
\((C_{q,\epsilon},S_q)\). Thus `maximum_normalized_noise_gain` is the worst
scaled query standard-deviation gain per unit whitened observation
perturbation, while `rms_normalized_noise_gain` summarizes the average scaled
query variance.

The scale is admitted without implicit jitter. It must be positive definite
above the frozen absolute-plus-relative eigenvalue tolerance. The certificate
also records its eigenvalues, Cholesky factor, numerical condition number, and
the normalized covariance.

## Decision states

The caller must freeze a dimensionless `noise_gain_limit` before target outcomes
are opened. The certificate returns one of four states:

- `nonidentifiable`: the bound source certificate is nonidentifiable, regardless
  of the gain computed on its identifiable projection;
- `trivial_query`: the source query is exactly zero and is not treated as an
  admitted update;
- `identifiable_but_unstable`: identifiability passes, but the worst normalized
  noise gain exceeds the frozen limit plus numerical comparison tolerance; or
- `stably_estimable`: identifiability passes and the gain bound passes.

There is no universal gain limit. It depends on the registered query scale,
scientific loss, and source protocol. The limit is part of the artifact identity
so it cannot be changed after outcomes are inspected without producing a new
artifact.

## Example

```python
import numpy as np

from bayesian_phystwin.query_estimability_certificate_v1 import (
    QueryEstimabilityCertificateV1,
)
from bayesian_phystwin.query_identifiability_certificate_v2 import (
    QueryIdentifiabilityCertificateV2,
)

identifiability = QueryIdentifiabilityCertificateV2(
    physical_response_id=physical_response_id,
    observation_mapping_id=observation_mapping_id,
    nuisance_design_id=nuisance_design_id,
    query_id=query_id,
    whitened_physical_design=np.diag([1.0, 1e-6]),
    whitened_nuisance_design=np.empty((2, 0)),
    query_map=np.eye(2),
)

assert identifiability.nontrivially_identifiable

estimability = QueryEstimabilityCertificateV1(
    identifiability_certificate=identifiability,
    query_scale_id=query_scale_id,
    query_scale=np.eye(2),
    noise_gain_limit=100.0,
)

assert estimability.status.value == "identifiable_but_unstable"
assert estimability.maximum_normalized_noise_gain == 1e6
```

The query is identifiable in exact arithmetic, but the weak second physical
response amplifies unit whitened noise by a factor of one million. An
identifiability-only gate would accept the interpretation; the estimability gate
rejects it as numerically unstable.

Changing the query to `np.array([[1.0, 0.0]])` removes dependence on the weak
latent direction. The resulting one-dimensional query has unit normalized gain
under a unit query scale and can pass a limit above one.

## Coordinate interpretation

The gain spectrum is invariant under a consistent invertible change of query
coordinates when the query map and query scale are transformed together:

\[
B' = CB,
\qquad
S_q'=CS_qC^\top.
\]

The exact artifact identity still changes because the registered coordinates,
scale bytes, and source certificate differ. This is intentional provenance.

The gain is not invariant to changing the observation whitening, omitting a
nuisance direction, changing the latent rank tolerance, or replacing the query
scale. Those choices define the local statistical problem and must be frozen by
the study protocol.

## Relationship to calibration and deployment guards

This certificate is analytic and local. It complements rather than replaces:

- nonlinear-closure checks over the registered horizon;
- source-group proper-score and harmful-update guards;
- covariance calibration and coverage diagnostics;
- held-out object or session transport; and
- exact complete-belief fallback.

A stable noise gain explains why a local identifiable query is not excessively
ill-conditioned. It does not show that the physical or nuisance Jacobian is
correct, that the whitened noise model is calibrated, or that an accepted update
improves prospective decisions.

## Scientific boundary

A passing certificate establishes only local linear query stability under the
exact bound identifiability artifact, unit-covariance whitened observation noise,
query scale, tolerances, and source-frozen gain limit. It does not establish:

- a unique data-generating or physical cause;
- correctness of the physical response, nuisance model, or whitening;
- global nonlinear or trajectory-wide identifiability;
- calibrated posterior uncertainty;
- provider competence or observation validity;
- unseen-object or independent-session transfer;
- prospective intervention transport;
- deployment safety; or
- downstream Causal4D benefit.

A `stably_estimable` result is necessary evidence for a future stability-aware
physical-query gate, not sufficient evidence for a physical-state claim or
deployment.
