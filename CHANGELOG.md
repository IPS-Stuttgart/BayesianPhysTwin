# Changelog

All notable user-visible changes to Bayesian PhysTwin are recorded here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
version numbers follow [Semantic Versioning](https://semver.org/) where the
pre-1.0 compatibility rules in [SUPPORT.md](SUPPORT.md) apply.

## [Unreleased]

### Added

- A NumPy-only, versioned Causal4D belief-provider surface for immutable robust
  Bayesian endpoint inference without downstream experiment-module imports.
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

### Changed

- The package now installs exactly one executable, `bpt`. Stable operations and
  research workflows are reached through grouped routes and lazy dispatch.
- Command help, documentation, and installed-artifact tests now distinguish
  stable interfaces, current experiments, non-promotable diagnostics, and
  archived reproduction paths.

### Removed

- The 79 top-level `bpt-*` console scripts. Frozen releases and historical
  manifests retain their original command strings; `bpt commands migrate`
  reports the current grouped invocation.

## Historical development

This changelog was introduced after the `0.4.0` development line had already
been established. Earlier changes remain documented by the Git history,
versioned experiment records, release tags, and frozen evidence manifests; they
are not reconstructed here as retrospective release notes.
