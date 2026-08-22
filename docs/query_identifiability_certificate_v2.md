# Query identifiability certificate v2

## Purpose

`bayesian_phystwin.query_identifiability_certificate_v2` answers a narrower and
more defensible question than full latent-state observability:

> Is the registered physical query determined by the observation after every
> declared nuisance direction has been removed?

A predictive residual can be useful without identifying a correction to the
latent physical state. Conversely, a physically relevant query can be
identifiable even when the complete latent state is not. The certificate makes
that distinction explicit and content-addresses the exact local linear problem,
numerical tolerances, factorization, and result.

The module is an experimental direct import. It does not add symbols to the
package root or to `bayesian_phystwin.v1`.

## Local model

Work in a common whitened observation coordinate system. Let

\[
\widetilde r = Xz + N\nu + \widetilde\epsilon,
\qquad
\Delta q = Bz,
\]

where:

- \(z\) contains the declared reachable physical-state coefficients;
- \(X\) is the whitened physical observation design;
- \(\nu\) contains all declared competing nuisance coefficients;
- \(N\) is the whitened nuisance design; and
- \(B\) maps the same physical coefficients to the registered query.

In the notation used by the guarded state-update derivation, one may have

\[
X=\widetilde H_xL_x,
\qquad
B=J_qL_x,
\]

but the certificate deliberately accepts the already assembled matrices. It
therefore binds the actual numerical linearization used by the caller rather
than reconstructing it from labels.

Let \(P_N\) be the orthogonal projector onto \(\operatorname{col}(N)\), and
define the nuisance-residualized physical design

\[
A=(I-P_N)X.
\]

Only \(A\) can distinguish physical response from the declared nuisance model.

## Proposition: query identifiability modulo nuisance

For finite-dimensional real matrices \(A\) and \(B\) with the same latent
column dimension, the following statements are equivalent:

1. every two latent perturbations that produce the same nuisance-residualized
   observation produce the same registered query;
2. \(\ker(A)\subseteq\ker(B)\);
3. \(\operatorname{row}(B)\subseteq\operatorname{row}(A)\);
4. there exists a linear operator \(M\) such that \(B=MA\); and
5. \(\operatorname{rank}([A;B])=\operatorname{rank}(A)\), where the matrices are
   stacked by rows.

Thus the query is identifiable modulo the declared nuisance model if and only
if any physical direction invisible to \(A\) is also irrelevant to \(B\).
Full latent-state identifiability is sufficient but not necessary.

### Proof sketch

Two latent perturbations \(z_1,z_2\) have the same residualized observation
exactly when \(z_1-z_2\in\ker(A)\). Their query values agree exactly when
\(z_1-z_2\in\ker(B)\), which gives the kernel-inclusion condition.

For finite-dimensional Euclidean spaces,
\(\ker(A)^\perp=\operatorname{row}(A)\). Taking orthogonal complements converts
\(\ker(A)\subseteq\ker(B)\) into
\(\operatorname{row}(B)\subseteq\operatorname{row}(A)\). Row-space inclusion is
equivalent to every row of \(B\) being a linear combination of rows of \(A\),
which is exactly the existence of \(M\) with \(B=MA\). Adding rows from \(B\)
then cannot increase rank, establishing the final equivalence.

## Numerical certificate

The implementation computes an SVD basis for the nuisance column space, forms
\(A\), and uses the tolerance-truncated pseudoinverse \(A^\dagger\). It records

\[
M=BA^\dagger,
\qquad
E=B-MA=B(I-A^\dagger A),
\]

and the normalized factorization residual

\[
\eta_{\mathrm{id}}
 =
 \frac{\lVert E\rVert_F}
      {\max(\lVert B\rVert_F,\operatorname{tiny}_{64})}.
\]

In exact arithmetic, \(E=0\) is equivalent to every statement in the
proposition. Numerically, the certificate returns:

- `identifiable` when the Frobenius residual is no larger than the frozen
  absolute-plus-relative identifiability bound;
- `nonidentifiable` when query energy remains in the numerical null space of
  the residualized physical design; or
- `trivial_query` when \(B\) is exactly zero.

The result also records:

- nuisance, physical, and query singular values and numerical ranks;
- the nuisance projector and residualized physical design;
- the factor operator \(M\) and factorization residual \(E\);
- physical nullity;
- the rank increment caused by query-visible null directions;
- Frobenius and spectral residual norms; and
- the fraction of query Frobenius energy explained through \(A\).

All input and derived arrays are copied into immutable `float64` storage. The
artifact identity binds their shapes, dtypes, byte-level SHA-256 digests,
upstream identities, tolerances, diagnostics, metadata, semantics, and claim
boundary.

## Example

```python
import numpy as np

from bayesian_phystwin.query_identifiability_certificate_v2 import (
    QueryIdentifiabilityCertificateV2,
)

certificate = QueryIdentifiabilityCertificateV2(
    physical_response_id=physical_response_id,
    observation_mapping_id=observation_mapping_id,
    nuisance_design_id=nuisance_design_id,
    query_id=query_id,
    whitened_physical_design=np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    ),
    whitened_nuisance_design=np.empty((2, 0)),
    query_map=np.array([[3.0, 0.0]]),
)

assert certificate.nontrivially_identifiable
assert np.allclose(
    certificate.factor_operator
    @ certificate.residualized_physical_design,
    certificate.query_map,
)
```

The second latent coefficient is not observable, but it does not affect the
query. The query is therefore identifiable even though the full two-dimensional
latent state is not.

Changing the query to `np.array([[0.0, 1.0]])` produces a nonidentifiable
certificate because the query then depends entirely on the hidden direction.

## Nuisance interpretation

The nuisance design must include every competing direction that the scientific
claim intends to rule out. Examples include:

- camera-frame or readout bias;
- global translation or gauge directions;
- declared timing, gain, or controller-frame nuisance modes; and
- locally linearized discrepancy directions when they are treated as a
  competing explanation.

Omitting a plausible nuisance mode can make a physical query appear identifiable
only because the alternative explanation was absent from the test. Adding an
unsupported nuisance mode can remove real physical information. The artifact
therefore binds the exact nuisance design rather than only a human-readable
name.

## Coordinate and basis invariance

The exact kernel-inclusion decision is invariant under:

- an invertible change of nuisance coefficients, because it preserves
  \(\operatorname{col}(N)\); and
- an invertible change of latent coordinates applied consistently to both
  \(X\) and \(B\).

The tests exercise both invariances on well-conditioned examples. The artifact
identity nevertheless changes when input bytes change, even when two nuisance
bases span the same subspace. This is intentional provenance, not a claim that
the scientific decision changed.

The finite-tolerance ranks and the normalized Frobenius residual are numerical
diagnostics, not coordinate-free scalars under arbitrary nonorthogonal
rescaling. A registered study must therefore freeze whitening, latent
coordinates, matrix construction, and tolerances before target outcomes are
opened. Orthogonal basis changes preserve the reported norms up to floating
point error; general invertible changes preserve only the exact mathematical
zero/nonzero property.

## Relationship to `IdentifiabilityReportV1`

`IdentifiabilityReportV1` records physically reachable modes retained after a
declared state-versus-bias distinguishability analysis. This certificate adds a
complementary query-level condition: whether the registered query vanishes on
every residualized physical null direction.

The two artifacts answer different questions:

- a mode report characterizes which physical coefficient directions are
  retained; and
- the query certificate proves, for the supplied local matrices and tolerance,
  whether the query factors through the residualized observation.

A physical-state candidate may bind the certificate artifact ID as its
identifiability evidence only when the candidate-construction protocol
explicitly declares these semantics. The hash field alone does not validate the
scientific meaning of an arbitrary artifact.

## Scientific boundary

A passing certificate establishes local linear query identifiability only under
the exact supplied matrices, whitening, nuisance span, coordinates, and
numerical tolerances. It does not establish:

- a unique data-generating or physical cause;
- correctness of the physical response or nuisance model;
- global nonlinear or trajectory-wide identifiability;
- provider competence or observation validity;
- uncertainty calibration or unseen-object transfer;
- prospective intervention transport;
- deployment safety; or
- downstream Causal4D benefit.

Those claims require their own frozen evidence and guards. A failed certificate
is a direct reason to reject the physical-state interpretation of the registered
query or to fall back to a nonphysical discrepancy/baseline belief; a passing
certificate is necessary evidence under the declared model, not sufficient
evidence for deployment.