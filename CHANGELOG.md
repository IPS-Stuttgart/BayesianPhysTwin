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
- A lazy `bpt experiment list|describe|run` registry for non-stable research
  commands.
- Wheel and source-distribution checks that enforce the single executable
  boundary.

### Changed

- Stable command-line operations now use grouped `bpt` routes exclusively.

### Removed

- All 79 legacy `bpt-*` console-script entry points. Historical command strings
  remain valid provenance records but are no longer executable aliases.

## Historical development

This changelog was introduced after the `0.4.0` development line had already
been established. Earlier changes remain documented by the Git history,
versioned experiment records, release tags, and frozen evidence manifests; they
are not reconstructed here as retrospective release notes.
