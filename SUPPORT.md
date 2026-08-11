# Support and compatibility policy

This document defines the supported Python runtime and the compatibility
contract between Bayesian PhysTwin, Prob4D, and Causal4D. Reproducible
experiments have stricter requirements than ordinary development installations
and must continue to record exact repository revisions and artifact digests.

## Python support

- The package metadata permits Python `>=3.10`.
- The required CI matrix tests the core contracts and full suite on Python
  `3.10`, `3.12`, and `3.14`.
- Other Python 3 releases at or above 3.10 are best-effort until they appear in
  the required CI matrix. A successful installation alone is not a support
  guarantee.
- Support applies to the latest patch release of each listed Python minor
  version. Optional CUDA, vision, graph, and external-model paths may impose
  narrower constraints through their upstream dependencies.

Dropping a tested Python minor version requires a documented compatibility
change. Adding a newly released Python minor version is not complete until the
core contracts, full test suite, wheel build, source-distribution build, and
installed-artifact smoke tests pass for it.

## Ecosystem compatibility table

The normative current development ranges and exact provider/schema boundaries
for BayesianPhysTwin, Prob4D, and Causal4D are published in
[`docs/ecosystem_compatibility_v1.md`](docs/ecosystem_compatibility_v1.md) and
the installed machine-readable resource
`bayesian_phystwin/contract_data/ecosystem_compatibility_v1/table.json`.

The current package lines are:

- `bayesian-phystwin>=0.4,<0.5`;
- `prob4d>=0.4,<0.5`; and
- `causal4d>=0.5,<0.6`.

These ranges express development interoperability only. They are not experiment
locks and do not establish accuracy, calibrated uncertainty, provider
competence, physical-query benefit, intervention benefit, or state of the art.
Claim-bearing runs must bind exact repository revisions, dependency resolver
input, provider attestations, and artifact digests.

The resource is validated by
`bayesian_phystwin.ecosystem_compatibility_v1`, which rejects unknown fields,
ambiguous JSON, noncanonical version ranges, provider/schema drift, coerced
Boolean flags, and any weakening of the exact-evidence boundary.

## Causal4D provider compatibility

`bayesian_phystwin.causal4d_provider_v1` remains the frozen scientific
compatibility facade for historical consumers. Current Causal4D integration is
defined by its versioned public-module registry, which also admits the
production request-complete replay surface
`bayesian_phystwin.causal4d_provider_v2` and the explicitly lifecycle-labelled
artifact, belief, graph, public-diagnostic, and tree-block modules recorded in
the ecosystem compatibility table.

- Bayesian PhysTwin `0.4.x` provides versioned provider APIs 1 and 2.
- Causal4D `0.5.x` may depend on `bayesian-phystwin>=0.4,<0.5`.
- Backward-compatible operations and capabilities may be added within the
  `0.4.x` line.
- Removing an operation, changing required semantics, units, shapes, failure
  behavior, or artifact interpretation requires a new provider module/API
  version and a new compatibility minor line.
- Frozen experiments must install exact Git revisions and verify recorded
  artifact hashes. The development version range never replaces experiment
  locks.

The normative provider-v1 details are maintained in
[`docs/causal4d_provider_v1.md`](docs/causal4d_provider_v1.md). Provider
manifests and matching local Causal4D contracts remain authoritative for
module-specific capability admission.

The fixed endpoint surface
`bayesian_phystwin.causal4d_belief_provider_v1` remains the compatibility
boundary for frozen discrepancy-endpoint consumers. The additive
`causal4d_belief_provider_v2` exposes evidence-weighted endpoint model averaging
and horizon-dependent model-based covariance. Adopting provider v2 is an
explicit consumer decision; it does not change provider v1 and does not imply
that the raw covariance is prospectively calibrated.

## Prob4D provider compatibility

Historical and exploratory observations may use the frozen
`prob4d.provider_v1` interface. Claim-bearing admission uses
`prob4d.provider_v2`, provider API version 2, together with a complete
provider-attestation schema version 1, causal-stream schema version 2, and the
artifact schema versions listed in the ecosystem compatibility table.

BayesianPhysTwin independently validates provider-v2 manifests, calibration
identities, covariance semantics, and runtime revision evidence without
importing Prob4D. A package-range match or successful observation parse does not
by itself authorize a physical update or a scientific claim.

The Prob4D provider manifests retain the historical source-repository identity
`FlorianPfaff/Prob4D` for content-addressed compatibility, while the maintained
repository is `IPS-Stuttgart/Prob4D`. This historical identity must not be
silently rewritten inside frozen manifests.

## Command-line compatibility

The current package installs one executable, `bpt`. Direct grouped routes are
the supported operational interface. Research commands remain available only
through their lifecycle catalogs:

```text
bpt experiment ...
bpt diagnostic ...
bpt archive ...
```

Historical `bpt-*` executable names are retained as registry metadata, not as
installed aliases. Use `bpt commands migrate LEGACY_ALIAS` to obtain the current
grouped invocation. Frozen releases and tags preserve the executable surface
with which their artifacts were created, and historical manifest command
strings remain immutable provenance.

Adding a command must not add another `[project.scripts]` entry. A new stable
route requires installed wheel and source-distribution coverage. Reclassifying
an experiment, diagnostic, or archived command requires an explicit owner and
documentation update.

## Public and experimental interfaces

Versioned artifact schemas, the Causal4D provider modules, the Prob4D
claim-bearing validation boundary, and commands exercised by installed-artifact
CI are supported interfaces. Underscore-prefixed modules, research scripts,
unregistered experiment entry points, and undocumented implementation details
are internal and may change without compatibility promises.

Because the project is pre-1.0, broader APIs may still evolve. Where practical,
a supported interface is deprecated for at least one compatibility line before
removal. Immediate fail-closed changes remain permitted when required to
correct causal leakage, provenance ambiguity, unsafe artifact loading, or
invalid scientific claims.

The exact package-root export surface for the `0.4.x` line is retained in
[`api/root-public-api-v0.4.json`](api/root-public-api-v0.4.json). It is a drift
ratchet for historical convenience imports, not a broader support promise.

The preferred ecosystem integration namespace is `bayesian_phystwin.v1`. Its
exact ordered export surface is retained in
[`api/versioned-public-api-v1.json`](api/versioned-public-api-v1.json). Changes
that remove or reinterpret those exports require a new versioned namespace;
research-only functionality should remain in explicit modules rather than
expanding `v1` without demonstrated consumer need.

Both surfaces are checked by
[`tools/quality/check_public_api.py`](tools/quality/check_public_api.py) and
shipped in the source distribution. See
[`docs/public_api_policy.md`](docs/public_api_policy.md) for the complete policy.

## Scientific release boundary

Runtime support, package compatibility, green CI, valid distributions, and
content-addressed integration artifacts are engineering evidence. They do not
promote an empirical result or authorize a deployment claim.

Every release note that cites the full-22 Bayesian-anchor result must follow
[`docs/phystwin_release_claim_v1.md`](docs/phystwin_release_claim_v1.md) and
retain all of these companion boundaries:

- the simple last-residual comparator is the principal matched deterministic
  reference and is marginally better on equal-case track error;
- the exact-mean covariance-only result changes the frozen Gaussian score while
  track and Chamfer outputs remain exactly unchanged, remains retrospective
  development evidence, and carries a `3.10×` interval-width cost;
- raw posterior covariance remains severely undercalibrated, while conformal
  coverage is a separate width-bearing result under its stated assumptions; and
- independent real-provider and independent-object transfer remain unconfirmed.

The canonical machine-readable companion is the paper repository's
[`evidence/bayesian_phystwin/bpt-release-synthesis-v1/summary.json`](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/blob/main/evidence/bayesian_phystwin/bpt-release-synthesis-v1/summary.json).
A compatibility-table match, accepted golden-path fixture, or exact fallback
record must never be described as accuracy, calibration, transfer, safety, or
state-of-the-art evidence.

## Reporting problems

Report reproducible defects through the repository issue tracker. Include the
Python version, Bayesian PhysTwin revision or package version, provider manifest
when relevant, operating system, optional dependency set, and the smallest
artifact or command that reproduces the problem. Do not attach restricted or
third-party data unless its terms permit redistribution.
