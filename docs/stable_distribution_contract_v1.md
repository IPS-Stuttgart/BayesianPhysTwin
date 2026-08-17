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
  `protocols/`, and `docs/`;
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

The 2026-08-14 consolidated material-backend change adds two reviewed transport
implementations, one canonical registry and CLI boundary, documentation, and
tests. The successful JAX-FEM/Genesis candidate already used 539 of the former
540 permitted wheel members. Combining the second transport and canonical
registry therefore raises the wheel-member ratchet from 540 to 550 and the
regular sdist-member ratchet from 1250 to 1270. Compressed-size limits,
isolated-import rules, API manifests, console scripts, and the supported
self-test list remain unchanged.

The repository test suite is broader than the release self-test subset and may
use CI helpers, workflow files, private operational state, optional dependencies,
or large research fixtures that are intentionally not publication artifacts.
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
