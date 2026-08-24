# Multi-action query identifiability v1

## Purpose

A residual observed under one action can remain indistinguishable from gauge,
camera bias, timing error, or predictive discrepancy. Several physically
different actions can nevertheless separate query-relevant physical directions
when their observation designs are considered jointly.

`MultiActionQueryIdentifiabilityCertificateV1` is a target-closed experimental
instrument for that question. It stacks action-specific prospective physical
designs and transported query maps, while accepting one **joint nuisance
design** over all observation rows. Shared nuisance coefficients therefore stay
shared across actions instead of being duplicated as independent per-action
errors.

The module lives in `bayesian_phystwin_experiments` and is not part of the
stable wheel or public API.

## Local model

For registered actions $a\in\mathcal A$, let

$$
\widetilde r_a = X_a z + N_a \nu + \widetilde\epsilon_a,
\qquad
\Delta q_a = B_a z.
$$

The caller supplies $X_a$ and the already transported query map $B_a$ in one
common latent coordinate system. The complete observation and query maps are

$$
X_{\mathcal A}
=
\begin{bmatrix}
X_{a_1}\\
\vdots\\
X_{a_K}
\end{bmatrix},
\qquad
B_{\mathcal A}
=
\begin{bmatrix}
B_{a_1}\\
\vdots\\
B_{a_K}
\end{bmatrix}.
$$

A separately supplied joint nuisance matrix $N_{\mathcal A}$ has the same
stacked observation rows. It may represent action-local nuisance coefficients,
shared camera or gauge variables, or a mixture of both.

After nuisance residualization,

$$
A_{\mathcal A}
=
\left(I-P_{N_{\mathcal A}}\right)X_{\mathcal A},
$$

the stacked registered queries are locally identifiable exactly when

$$
\ker(A_{\mathcal A})\subseteq\ker(B_{\mathcal A}).
$$

This is the existing query-identifiability proposition applied to the stacked
matrices. The wrapper reuses `QueryIdentifiabilityCertificateV2` rather than
implementing a second numerical criterion.

## Diagnostics

The artifact records:

- the complete joint certificate;
- each action's single-action certificate status;
- a leave-one-action-out certificate that removes only that action's
  observation rows while retaining the complete registered query stack;
- the loss of identifiable query energy caused by removing each action; and
- whether the full query is identifiable while no single action is
  individually sufficient.

The last condition is reported as `requires_multiple_actions`. It is evidence
that the supplied actions are complementary under the declared local model. It
is not evidence that the actions are safe, correctly modelled, or optimal.

## Example

```python
import numpy as np

from bayesian_phystwin_experiments.multi_action_query_identifiability_v1 import (
    ActionIdentifiabilityBlockV1,
    MultiActionQueryIdentifiabilityCertificateV1,
)

blocks = (
    ActionIdentifiabilityBlockV1(
        action_id="bend",
        physical_response_id=physical_response_id,
        observation_mapping_id=observation_mapping_id,
        query_transport_id=bend_query_transport_id,
        whitened_physical_design=np.array([[1.0, 0.0]]),
        transported_query_map=np.array([[1.0, 1.0]]),
    ),
    ActionIdentifiabilityBlockV1(
        action_id="stretch",
        physical_response_id=physical_response_id,
        observation_mapping_id=observation_mapping_id,
        query_transport_id=stretch_query_transport_id,
        whitened_physical_design=np.array([[0.0, 1.0]]),
        transported_query_map=np.array([[1.0, 1.0]]),
    ),
)

certificate = MultiActionQueryIdentifiabilityCertificateV1(
    latent_coordinates_id=latent_coordinates_id,
    whitening_id=whitening_id,
    joint_nuisance_design_id=nuisance_design_id,
    joint_query_id=query_id,
    action_blocks=blocks,
    joint_whitened_nuisance_design=np.empty((2, 0)),
)

assert certificate.identifiable
assert certificate.requires_multiple_actions
```

Action blocks must be sorted by `action_id`, use identical latent coordinates,
and be frozen before target outcomes are opened.

## Relationship to cross-action transport

This certificate is a source-side admission condition. It asks whether the
registered transported queries factor through the stacked, nuisance-residualized
local observation model.

`CrossActionTransportResultV1` remains the prospective empirical test. It asks
whether a source-inferred physical candidate actually improves held-out actions
relative to physical fallback, discrepancy persistence, and the matched
deterministic comparator. A passing certificate cannot rescue a failed
transport result, and a positive transport result cannot repair an omitted
nuisance direction in the certificate.

## Scientific boundary

A passing artifact establishes only local linear query identifiability under the
exact matrices, whitening, nuisance span, latent coordinates, query maps, and
numerical tolerances. It does not establish:

- a unique physical or data-generating cause;
- correctness of the physical response, transport, or nuisance model;
- global nonlinear or trajectory-wide identifiability;
- safe action execution;
- uncertainty calibration or unseen-object transfer;
- real Prob4D provider competence;
- completed Causal4D physical evidence;
- deployment safety; or
- deformable-object state of the art.
