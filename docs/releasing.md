# Release candidate and publication policy

BayesianPhysTwin separates **release-candidate evidence** from **publication**.
Building a valid wheel and source distribution is a necessary packaging result;
it is not permission to publish, a scientific result, or evidence that a
physical method is safe or effective.

## Release preparation

Prepare the release on an ordinary pull request before creating a tag:

1. Select the compatibility line. Patch releases in `0.4.x` must preserve the
   frozen root export surface in `api/root-public-api-v0.4.json`. A future `0.5`
   line requires its own API snapshot and documented replacement imports.
2. Update the literal version in both `pyproject.toml` and `CITATION.cff`.
3. Move the relevant entries from `CHANGELOG.md`'s `Unreleased` section into a
   dated version section. Keep scientific evidence changes distinct from
   packaging, infrastructure, and compatibility changes.
4. Run the normal test workflow, cross-repository compatibility checks that are
   required for the changed interfaces, security scanning, and the release
   candidate workflow.
5. Review the generated `release-evidence.json`, wheel, source distribution,
   CycloneDX SBOM, build-environment record, and workflow summary.

The release-candidate workflow checks out the exact pull-request head or tag
commit, binds `SOURCE_DATE_EPOCH` to that commit, installs an exact release
build stack, builds without an implicit isolated toolchain, applies strict Twine
validation, resolves and audits runtime dependencies, and emits a
content-addressed evidence record.

## Evidence boundary

`tools/release/build_release_evidence.py` requires and binds:

- one wheel and one source distribution;
- matching project, citation, wheel, and source-distribution versions;
- the expected `vX.Y.Z` tag when executed from a tag;
- required license, typing, API-policy, and release-tool members;
- canonical archive paths and a link-free source distribution;
- the exact source revision and source contract hashes;
- the complete recorded Python build environment;
- the runtime CycloneDX SBOM; and
- SHA-256 and byte counts for every retained release artifact.

The generated record deliberately states that it is build and provenance
evidence only. It does not publish to PyPI, create a GitHub release, modify a
scientific claim, or authorize deployment.

## Tagging

Create an annotated tag only after the release preparation pull request is on
`main` and every required check is green:

```bash
git tag -a vX.Y.Z -m "BayesianPhysTwin X.Y.Z"
git push origin vX.Y.Z
```

The tag name must exactly equal `v` followed by the project version. A mismatched
tag fails before a release candidate can be accepted. Preserve the tag workflow's
artifact bundle and its `evidence_id` with any eventual release record.

## Publication

This repository does not currently publish from the release-candidate workflow.
A future PyPI publication workflow should be a separate, minimal tag-only job
that:

- consumes or independently rebuilds and verifies the exact candidate artifacts;
- uses a protected `pypi` GitHub environment with independent approval;
- uses PyPI Trusted Publishing rather than a long-lived API token;
- grants `id-token: write` only to the publication job;
- publishes no artifact whose tag, version, digest, SBOM, or evidence identity
  differs from the reviewed candidate; and
- records the resulting PyPI provenance and GitHub release asset hashes.

Publication must never run on a pull request, branch push, schedule, or
self-hosted dataset runner.
