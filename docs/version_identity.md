# Release version identity

A source revision must not build a distribution with the same version as a
release tag that points to another commit. Without this invariant, a checkout of
`main` can produce bytes that identify themselves as an older release even
though the source, behavior, and evidence have changed.

BayesianPhysTwin therefore treats the project version as part of source
provenance:

- `pyproject.toml` and `CITATION.cff` must contain the same literal version;
- the canonical tag for that version is `v<version>`;
- an untagged final version is permitted only as a release candidate;
- once the canonical tag exists, only the tagged commit may retain that version;
- subsequent development must move immediately to the next `.dev0` version; and
- repository validation requires complete Git history so a shallow checkout
  cannot mistake an unfetched tag for an unused version.

The checker is available from a complete source checkout:

```bash
python tools/release/check_version_identity.py \
  --require-complete-history
```

An explicit tag validation additionally requires the canonical tag:

```bash
python tools/release/check_version_identity.py \
  --require-complete-history \
  --expected-tag "vX.Y.Z"
```

The existing Python quality ratchet runs the repository check on pull requests
and `main` from a full-history checkout. It passes the exact checked-out commit
rather than trusting the working tree. The release-candidate workflow separately
binds a tag push to its exact source revision and requires the tag name to equal
`v` followed by the project version before accepting release evidence.

The unit tests reproduce an annotated release tag, a post-release commit that
incorrectly reuses the old version, metadata disagreement, and the correct
transition to the next development version. A static integration test also keeps
the quality-ratchet invocation from being removed accidentally.

## Development-line rule

After publishing `vX.Y.Z`, the next ordinary commit must update both metadata
files to the next intended development version, normally
`X.Y.(Z+1).dev0`, and remove or defer any release date until a final release is
prepared. For example, development after `v0.4.0` uses `0.4.1.dev0`.

Before publishing the next release, replace the development version with the
final version, restore the release date in `CITATION.cff`, complete the changelog,
and follow the release-candidate procedure in [releasing.md](releasing.md).

## Evidence boundary

This check establishes distribution and source identity only. It does not alter
an estimator, artifact schema, experiment, scientific claim, compatibility
range, or deployment authorization.
