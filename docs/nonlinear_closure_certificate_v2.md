# Nonlinear closure certificate v2

## Purpose

A local physical-state or parameter correction should not be interpreted from a
Jacobian alone. The registered linearized query response must also agree with a
nonlinear replay over a source-frozen perturbation set.

The historical `NonlinearClosureV1` remains unchanged and records one aggregate
absolute/relative comparison for one replay. The additive
`NonlinearClosureCertificateV2` closes three claim-integrity gaps:

1. it binds the exact `PhysicalLinearizationV1`, perturbation set, query set,
   horizon assignment, replay arrays, and tolerances by content identity;
2. it evaluates every perturbation/query pair rather than allowing a large,
   well-modelled query to dominate one aggregate norm; and
3. it reports the exact worst perturbation, query, and horizon for failure
   localization.

The module is an experimental direct import:

```python
from bayesian_phystwin.nonlinear_closure_certificate_v2 import (
    NonlinearClosureCertificateV2,
)
```

It adds no package-root symbol, changes no V1 artifact, and is not inserted into
any frozen target or confirmation protocol.

## Frozen replay contract

For `P` registered perturbations and `Q` registered three-dimensional queries,
the caller supplies:

- baseline query values `q0` with shape `(Q, 3)`;
- linearized query values `q_lin` with shape `(P, Q, 3)`;
- nonlinear replay values `q_nl` with shape `(P, Q, 3)`;
- unique nonnegative perturbation and query indices;
- one nonnegative horizon index per query; and
- one absolute and relative tolerance per query.

The query count must equal the row count of the bound
`PhysicalLinearizationV1.query_state_jacobian`. Query and perturbation indices
are labels, not statistical replicates; the perturbation-set and query-set
content identities own their scientific construction and ordering.

For perturbation `p` and query `q`, define the predicted change

\[
  d_{pq}=\lVert q^{\mathrm{lin}}_{pq}-q^0_q\rVert_2
\]

and nonlinear remainder

\[
  e_{pq}=\lVert q^{\mathrm{nl}}_{pq}-q^{\mathrm{lin}}_{pq}\rVert_2.
\]

The admitted mixed absolute/relative allowance is

\[
  a_{pq}=a_q+r_q\max(d_{pq},d_{\min}),
\]

where `a_q` is the source-frozen absolute tolerance, `r_q` the source-frozen
relative tolerance, and `d_min` the positive prediction floor. Each query must
have a positive absolute or relative tolerance, so every allowance is strictly
positive.

The dimensionless local closure ratio is

\[
  c_{pq}=e_{pq}/a_{pq}.
\]

The certificate reports

\[
  c_{\max}=\max_{p,q} c_{pq},
\]

along with per-query and per-horizon maxima. The status is `locally_closed` only
when

\[
  c_{\max}\leq c_{\mathrm{limit}}+\epsilon_{\mathrm{cmp}},
\]

where both the limit and numerical comparison tolerance are part of the
artifact identity. Ordinary violations return `closure_violation`; this module
does not select a replacement belief or weaken the caller-owned exact fallback.

## Why one aggregate norm is insufficient

Suppose one query changes by `100 m`, a second changes by `1 mm`, and the second
has a `2 mm` nonlinear remainder. A global relative norm can be approximately
`2e-5` because the large first query dominates the denominator, even though the
small second query is locally unusable. V2 scores the two query rows separately,
so the second row cannot be rescued by the first.

This distinction matters for heterogeneous physical queries, mixed horizons,
and sparse perturbation directions. The certificate is deliberately
worst-case: a claim-bearing physical interpretation should not omit the one
registered query or perturbation on which its local model fails.

## Example

```python
import numpy as np

from bayesian_phystwin.nonlinear_closure_certificate_v2 import (
    NonlinearClosureCertificateV2,
)

certificate = NonlinearClosureCertificateV2(
    linearization=physical_linearization,
    perturbation_set_id=perturbation_set_id,
    query_set_id=query_set_id,
    perturbation_indices=np.asarray([10, 20]),
    query_indices=np.asarray([100, 200, 300]),
    horizon_indices=np.asarray([1, 1, 2]),
    baseline_query_m=baseline_query,
    linearized_query_m=linearized_queries,
    nonlinear_query_m=nonlinear_replays,
    absolute_tolerance_m=np.asarray([0.001, 0.001, 0.002]),
    relative_tolerance=np.asarray([0.10, 0.10, 0.15]),
    prediction_floor_m=1e-6,
    closure_ratio_limit=1.0,
)

if certificate.passes_closure_gate:
    candidate = proposed_complete_belief
else:
    candidate = exact_caller_owned_baseline

print(certificate.summary())
```

All input and derived arrays returned by `arrays()` are immutable. The
descriptor records their shape, dtype, and SHA-256 digest. Supplying an
`artifact_id` revalidates the complete content and fails closed on drift.

## Relationship to the admission ladder

Nonlinear closure is one conjunctive source-side condition. It complements but
does not replace:

1. provider support and provider-to-physical mapping;
2. query identifiability modulo declared nuisance;
3. stable estimability under the exact whitening and query scale;
4. source-group proper-score, harmful-update, and interval-width guards;
5. held-out object/session transport; and
6. held-out intervention transport for a physical-cause interpretation.

A passing certificate says that the fixed local linear model reproduces the
registered nonlinear replays within the frozen tolerance policy. It does not say
that the simulator, Jacobian, nuisance model, observation provider, or
perturbation distribution is correct.

## Scientific boundary

A `locally_closed` result establishes only local nonlinear replay agreement over
the exact bound physical linearization, perturbation set, query set, horizons,
replay arrays, and source-frozen tolerances. It does not establish:

- a unique physical or data-generating cause;
- global or trajectory-wide nonlinear validity;
- calibrated posterior uncertainty;
- provider competence or observation validity;
- unseen-object or independent-session transfer;
- held-out intervention transport;
- deployment safety;
- downstream Causal4D benefit; or
- state of the art.

Future claim-bearing integration should consume the exact certificate identity
and route every rejection to the complete caller-owned baseline belief. A new
perturbation set, query set, horizon mapping, tolerance, simulator revision, or
replay produces a new artifact and requires a separately frozen source decision.
