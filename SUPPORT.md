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

## Public and experimental interfaces

Versioned artifact schemas, the Causal4D provider module, and the six stable
routes documented in [`docs/command_line.md`](docs/command_line.md) are
supported interfaces. The `bpt experiment` dispatcher and registry format are
supported infrastructure, but individual experiment identifiers and their
arguments remain non-stable research interfaces.

Underscore-prefixed modules, unregistered research scripts, and undocumented
implementation details are internal and may change without compatibility
promises. Legacy `bpt-*` executables are not supported aliases; frozen command
strings remain provenance only.

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
