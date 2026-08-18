# Stable distribution contract v1

BayesianPhysTwin contains two deliberately different surfaces:

1. a small, supported integration boundary intended for ordinary wheel users;
2. a much larger research, protocol, and evidence surface retained in the source
   repository and source distribution for reproducibility.

The release process previously checked archive integrity, metadata, typing
markers, licensing, the SBOM, and source provenance. It did not place an
executable bound on wheel growth or prove that the stable namespaces imported
from the built wheel without loading optional research dependencies. It also did
not state which tests in the broad source distribution form the supported sdist
self-test boundary.

`release/stable_distribution_contract_v1.json` closes those gaps for the 0.4
compatibility line. The contract is prospective release infrastructure. It does
not alter an estimator, a frozen protocol, an artifact schema, or an existing
scientific result.

## Wheel contract

The wheel gate verifies all of the following without extracting the archive:

- a maximum compressed size and member count;
- required stable namespace and PEP 561 members;
- absence of repository-only directories such as `tests/`, `scripts/`,
  `protocols/`, `docs/`, and the sibling source-only
  `bayesian_phystwin_experiments` namespace;
- exactly one declared console script, `bpt`;
- exact `__all__` parity with the root, portable-artifact, and guarded-inference
  API manifests; and
- isolated imports directly from the candidate wheel with no graph, vision,
  data, experiment, or heavy numerical dependency leakage.

The numerical limits are ratchets, not performance claims. Increasing either
limit requires an explicit contract change and review. A future 0.5 migration
may reduce the wheel substantially; version 1 prevents accidental growth while
that migration is prepared.

## Source-distribution contract

The sdist gate verifies:

- a maximum compressed size and regular-member count;
- one canonical archive root and no symbolic or hard links;
- the distribution contract, checker, documentation, and public API manifests;
  and
- an explicit supported self-test subset.

The 2026-08-14 consolidated material-backend change added two reviewed
transport implementations and one canonical registry and CLI boundary.
Subsequent reviewed backend integrations brought current `main` to 556 wheel
members, 2,197,884 compressed wheel bytes, and 1,290 regular sdist members. The
official MatPhys producer adds two wheel members and brings a reproducible build
to 2,208,784 bytes and 1,293 regular sdist members. This change therefore sets
the corresponding ratchets to 560 members, 2,220,000 bytes, and 1,300 regular
sdist members. The 3,100,000-byte sdist limit, isolated-import rules, API
manifests, console scripts, and supported self-test list remain unchanged.

Later reviewed integrations raised the ratchets to 580 wheel members,
2,280,000 compressed wheel bytes, 1,350 regular sdist members, and 3,200,000
compressed sdist bytes. The Genesis MPM source qualification adds two backend
gate modules. A clean candidate build contains 576 wheel members and 2,283,461
compressed wheel bytes; its source distribution contains 1,348 regular members
and 3,176,535 compressed bytes. The wheel-size ratchet is therefore raised
narrowly to 2,290,000 bytes. Member-count and source-distribution limits remain
unchanged. This packaging adjustment does not promote Genesis: the frozen
source-value gate rejected it and retained the incumbent through exact fallback.

The repository test suite is broader than the release self-test subset and may
use CI helpers, workflow files, private operational state, optional dependencies,
or large research fixtures that are intentionally not publication artifacts.
Source-only experiment implementations and their dedicated tests likewise stay
in the repository rather than the stable wheel or sdist.
The sdist therefore does **not** claim that every repository test can be run from
an unpacked source release. The supported self-test files are listed literally
in the contract and must all be present.

After installing the candidate wheel, a reviewer can run the supported subset
from an unpacked sdist with:

```bash
python -m pytest -q \
  tests/test_versioned_api_v1.py \
  tests/test_inference_v1.py \
  tests/test_public_api_manifest.py \
  tests/test_spd_system.py \
  tests/test_spd_system_adversarial.py \
  tests/test_stable_distribution_contract.py
```

## Release evidence

`tools/release/build_release_evidence.py` validates this contract before creating
release evidence. The generated record binds:

- the contract path and SHA-256 digest;
- observed and permitted wheel/sdist sizes and member counts;
- console scripts;
- API-manifest parity;
- isolated-import reports; and
- the supported sdist self-test list.

A valid report establishes distribution-surface conformance only. It is not a
PyPI publication, scientific result, uncertainty-calibration result, provider
competence result, or deployment approval.
