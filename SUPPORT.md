# Support and compatibility policy

This document defines the supported Python runtime and the compatibility
contract between Bayesian PhysTwin and Causal4D. Reproducible experiments have
stricter requirements than ordinary development installations and must continue
to record exact repository revisions and artifact digests.

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

The distribution includes `bayesian_phystwin/py.typed`. Public annotations are
therefore available to PEP 561 consumers from both wheel and source-distribution
installations.

## Causal4D provider compatibility

`bayesian_phystwin.causal4d_provider_v1` is the supported integration surface
for Causal4D.

- Bayesian PhysTwin `0.4.x` provides provider API/schema version 1.
- Upgradeable Causal4D development environments may depend on
  `bayesian-phystwin>=0.4,<0.5`.
- Backward-compatible operations and capabilities may be added within the
  `0.4.x` line.
- Removing an operation, changing required semantics, units, shapes, failure
  behavior, or artifact interpretation requires a new provider module/API
  version and a new compatibility minor line.
- Frozen experiments must install exact Git revisions and verify recorded
  artifact hashes. The development version range never replaces experiment
  locks.

The normative provider details are maintained in
[`docs/causal4d_provider_v1.md`](docs/causal4d_provider_v1.md).

## Prob4D repository identity compatibility

The canonical active producer is `IPS-Stuttgart/Prob4D`. Content-addressed
artifacts released before the repository transfer record
`FlorianPfaff/Prob4D`; that descriptor is retained as a frozen compatibility
identity and must not be rewritten.

Both identities are accepted only at the versioned Prob4D observation boundary.
All causal lineage, covariance, metric-anchor, revision, and digest checks remain
mandatory. For provider-v2 artifacts, the observation descriptor and embedded
provider manifest must declare the same supported repository identity. A mixed
canonical/frozen declaration fails closed.

The exact migration contract is documented in
[`docs/repository_identity_migration.md`](docs/repository_identity_migration.md).

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

Versioned artifact schemas, the Causal4D provider module, and commands exercised
by installed-artifact CI are supported interfaces. Underscore-prefixed modules,
research scripts, unregistered experiment entry points, and undocumented
implementation details are internal and may change without compatibility
promises.

Because the project is pre-1.0, broader APIs may still evolve. Where practical,
a supported interface is deprecated for at least one compatibility line before
removal. Immediate fail-closed changes remain permitted when required to correct
causal leakage, provenance ambiguity, unsafe artifact loading, or invalid
scientific claims.

## Reporting problems

Report reproducible defects through the repository issue tracker. Include the
Python version, Bayesian PhysTwin revision or package version, provider manifest
when relevant, operating system, optional dependency set, and the smallest
artifact or command that reproduces the problem. Do not attach restricted or
third-party data unless its terms permit redistribution.
