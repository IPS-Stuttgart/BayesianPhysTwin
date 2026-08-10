# Stable Prob4D provider-v2 bridge

New BayesianPhysTwin integrations should resolve claim-bearing Prob4D artifacts
through the dedicated bridge module:

```python
from bayesian_phystwin.prob4d_api_v2 import (
    inspect_prob4d_api_v2,
    load_claim_bearing_tree_sparse_prob4d,
)

compatibility = inspect_prob4d_api_v2()
validated = load_claim_bearing_tree_sparse_prob4d("observation-envelope.json")
```

The bridge imports `prob4d.api.v2` lazily. Importing BayesianPhysTwin therefore
does not make Prob4D a base dependency or load optional vision/GPU packages.
When a claim-bearing artifact is requested, the bridge fails closed unless the
installed producer exposes:

- stable API version 2;
- provider API version 2;
- provider-factor API version 2;
- project identity `github-repository-id:1295794737`; and
- the strict claim-bearing tree-sparse loader.

The current canonical navigation repository is `IPS-Stuttgart/Prob4D`.
Historical content-addressed observations may correctly retain
`FlorianPfaff/Prob4D`; the bridge validates the producer's transfer-safe project
descriptor rather than rewriting those frozen bytes.

## Compatibility boundary

The frozen `bayesian_phystwin.v1` API remains unchanged. The provider-v2 bridge
uses its own module because adding new names to that deliberately small surface
would silently broaden an existing compatibility contract. A future formal
BayesianPhysTwin API revision can adopt the bridge explicitly without changing
version 1.

## Ownership boundary

Prob4D owns provider-v2 validation, artifact integrity, causal lineage, gauge
priors, calibration identities, and tree-sparse loading. BayesianPhysTwin then
independently validates the evidence fields required by its physical update and
owns robust inference, regret guards, and exact physical fallback.

The legacy package-root and implementation-module routes remain available for
frozen compatibility. New integrations should not import
`prob4d.provider_v2_factors`, `prob4d.gauge`, or `prob4d.sim3` directly; the
stable producer boundary is `prob4d.api.v2`.

A successful compatibility inspection establishes only software and contract
compatibility. It does not establish observation accuracy, covariance
calibration, physical-query benefit, deployment safety, or state of the art.
