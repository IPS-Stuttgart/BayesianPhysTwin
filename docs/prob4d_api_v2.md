# Stable Prob4D provider-v2 bridge

New BayesianPhysTwin integrations should resolve claim-bearing Prob4D artifacts
through the dedicated bridge module:

```python
from bayesian_phystwin.prob4d_api_v2 import (
    inspect_prob4d_api_v2,
    inspect_prob4d_provider_v2_contract,
    load_claim_bearing_tree_sparse_prob4d,
)

compatibility = inspect_prob4d_api_v2()
conformance = inspect_prob4d_provider_v2_contract()
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

## Provider-v2 conformance corpus

`inspect_prob4d_provider_v2_contract()` additionally verifies the installed
advanced factor-contract corpus before ecosystem integration tests use it. The
probe requires:

- bundle `prob4d.provider_v2_factors.v1`;
- exact content identity
  `fe0374f46319287e3709497de9cbb73f7497286cf4f157f246096f2c352e4446`;
- exactly one valid `minimal` vector and ten adversarial vectors;
- minimal tree-prior identity
  `ddb97db5c953635eaa881c4d1b1fbe3e9508a72d0c0fb13a5d2a7f5727021dee`;
- portable structural stack identity
  `58621710b5b22a64163c47b4756f200cea13e56491d85a3852af96ec1cb0f4fb`;
  and
- numerical parity tolerances `atol=1e-12` and `rtol=1e-10`.

The probe calls Prob4D's installed corpus materializer and verifier, but does
not use its runtime-specific digest of derived floating-point bytes as a
compatibility condition. Supported NumPy or BLAS implementations may differ in
the final bits while satisfying the exact contract identities and numerical
parity checks.

This check is intentionally separate from claim-bearing admission.
`tree_sparse_explicit_gauge_prob4d` still independently validates the envelope,
causal lineage, row geometry, probabilities, gauge tree, calibration evidence,
and physical-update inputs. Passing the corpus probe cannot bypass those
consumer-owned checks.

## Compatibility boundary

The frozen `bayesian_phystwin.v1` API remains unchanged. The provider-v2 bridge
uses its own module because adding new names to that deliberately small surface
would silently broaden an existing compatibility contract. A future formal
BayesianPhysTwin API revision can adopt the bridge explicitly without changing
version 1.

The conformance probe is additive and optional for installations that do not
consume Prob4D. An installation that does consume the advanced provider-v2
factor path should run it before executing cross-repository integration or
release-capsule checks.

## Ownership boundary

Prob4D owns provider-v2 validation, artifact integrity, causal lineage, gauge
priors, calibration identities, and tree-sparse loading. BayesianPhysTwin then
independently validates the evidence fields required by its physical update and
owns robust inference, regret guards, and exact physical fallback.

The legacy package-root and implementation-module routes remain available for
frozen compatibility. New integrations should not import
`prob4d.provider_v2_factors`, `prob4d.gauge`, or `prob4d.sim3` directly; the
stable producer boundary is `prob4d.api.v2`.

A successful compatibility or corpus inspection establishes only software and
contract compatibility. It does not establish observation accuracy, covariance
calibration, physical-query benefit, deployment safety, or state of the art.
