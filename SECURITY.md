# Security policy

## Supported versions

Security fixes are applied to the current development line. Frozen research tags
and exact experiment revisions remain immutable for reproducibility; a security
issue affecting one of those revisions is documented and repaired in a new
version rather than silently rewriting the historical source.

| Version | Supported |
| --- | --- |
| Current `main` / latest release | Yes |
| Older development releases | Best effort |
| Frozen experiment revisions | Reproduced as recorded; repair in a successor |

## Reporting a vulnerability

Do not disclose a suspected vulnerability in a public issue, pull request, log,
or artifact.

Use GitHub's private vulnerability-reporting or security-advisory interface for
this repository. Include:

- the affected revision or package version;
- the smallest reproducible example;
- the expected and observed behavior;
- whether credentials, private data, self-hosted runners, artifact integrity, or
  arbitrary code execution are involved; and
- any temporary mitigation already applied.

Do not attach restricted datasets, credentials, access tokens, private model
weights, or third-party material unless its terms permit that disclosure.

## Security-sensitive areas

The following changes receive heightened review:

- GitHub Actions and self-hosted runner execution;
- artifact loading, path validation, archive extraction, and deserialization;
- content-addressed identities, evidence locks, and provenance validation;
- cross-repository provider admission;
- credentials, data roots, model caches, and external downloads; and
- code that changes fail-closed behavior or exact fallback.

Pull-request code must not receive private dataset roots, confirmation payloads,
SSH keys, package-publishing credentials, or a writable Docker socket merely to
run tests. Privileged empirical workflows should execute only reviewed exact
revisions and should keep dataset-bearing execution separate from general
pull-request validation.

## Response expectations

A report is first triaged for reproducibility and impact. Confirmed issues are
assigned a private repair plan and a disclosure boundary. Research provenance is
preserved: affected evidence is marked, successor artifacts receive new
identities, and historical result bytes are not rewritten.
