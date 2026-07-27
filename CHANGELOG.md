# Changelog

All notable user-visible changes to Bayesian PhysTwin are recorded here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
version numbers follow [Semantic Versioning](https://semver.org/) where the
pre-1.0 compatibility rules in [SUPPORT.md](SUPPORT.md) apply.

## [Unreleased]

### Added

- An MIT license for project-authored source code and documentation.
- Machine-readable software citation metadata in `CITATION.cff`.
- A Python and Causal4D provider compatibility policy in `SUPPORT.md`.
- A third-party source, model, checkpoint, dataset, and generated-artifact
  boundary in `THIRD_PARTY_NOTICES.md`.
- Package metadata and project links for licensing, citation, support, and this
  changelog.
- Distribution manifests and regression checks that keep release metadata
  present and version-consistent.
- A typed `bpt` command registry with lifecycle, optional-dependency, ownership,
  and removed-alias metadata.
- Grouped `experiment`, `diagnostic`, and `archive` catalogs plus a migration
  lookup for historical `bpt-*` command names.
- A frozen full-22 Bayesian-anchor reproduction capsule that binds the exact
  historical source revision, data manifest, protocol, expected metrics,
  two-stage source command, and `RunManifestV2` evidence bundle.
- A NumPy-only, versioned Causal4D belief-provider surface for immutable robust
  Bayesian endpoint inference without downstream experiment-module imports.
- Independent validation of self-contained Prob4D provider-v2 attestations,
  including the embedded manifest, calibration IDs, numerical modes, and runtime
  revision evidence.
- A strict claim-bearing Prob4D validation entry point for new prospective
  Prob4D-to-Bayesian-PhysTwin experiments while retaining provider-v1 reproduction.
- An always-executed Bayesian-PhysTwin and Causal4D consumer fixture for the
  cross-repository observation and lineage boundary.

### Changed

- The package now installs exactly one executable, `bpt`. Stable operations and
  research workflows are reached through grouped routes and lazy dispatch.
- Command help, documentation, and installed-artifact tests now distinguish
  stable interfaces, current experiments, non-promotable diagnostics, and
  archived reproduction paths.
- Prob4D causal-lineage validation now fails closed on any present but malformed
  provider-v2 attestation and reports a compact validated provider summary.
- Missing private-Prob4D credentials now fail trusted pull requests, `main`,
  scheduled, and manual three-repository runs instead of producing a green skip.
  External-fork pull requests still run the producer-neutral consumer fixture and
  explicitly report that the secret-backed producer gate was unavailable and no
  current-Prob4D evidence was admitted.
- Observation-belief metadata is now recursively immutable after canonical JSON
  validation, preventing nested mutation from changing an existing artifact digest.
- Grouped low-rank covariance statistics now use blockwise Cholesky/Woodbury solves
  without explicit covariance inverses or a dense all-factor-groups matrix.
- Fixed endpoint posteriors expose an explicit read-only `updated_mask`, and
  no-support summaries serialize JSON `null` rather than non-finite statistics.

### Removed

- The 79 top-level `bpt-*` console scripts. Frozen releases and historical
  manifests retain their original command strings; `bpt commands migrate`
  reports the current grouped invocation.
- The duplicate standalone gauge-aware workflow; the main test matrix already
  runs the same gauge-aware and prior-aware tests across its core and full-suite
  jobs.

## Historical development

This changelog was introduced after the `0.4.0` development line had already
been established. Earlier changes remain documented by the Git history,
versioned experiment records, release tags, and frozen evidence manifests; they
are not reconstructed here as retrospective release notes.
