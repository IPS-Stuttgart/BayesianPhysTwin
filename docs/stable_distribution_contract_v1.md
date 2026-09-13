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
- exact `__all__` parity with the root, portable-artifact, strict guarded-
  inference, and provider-neutral inference-session API manifests; and
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

The JAX-FEM source qualification adds two reviewed backend-gate modules. A
clean candidate build contains 580 wheel members and 2,324,144 compressed
wheel bytes; its source distribution contains 1,357 regular members and
3,229,248 compressed bytes. The corresponding ratchets are therefore raised
narrowly to 2,330,000 wheel bytes, 1,360 regular sdist members, and 3,240,000
sdist bytes. The wheel-member limit remains unchanged. This packaging change
does not promote JAX-FEM: its frozen linear small-strain quasistatic
source-value arm was rejected before outcome access, and exact fallback
preserved the incumbent.

The guarded MatPhys uncertainty experiment adds four reviewed implementation
modules while leaving the frozen DEFORM mean unchanged. Under the pinned
release toolchain, current `main` builds a 582-member, 2,335,468-byte wheel and
the candidate builds a 586-member, 2,355,401-byte wheel. The compressed-size
ratchet is therefore raised narrowly to 2,360,000 bytes; the 590-member limit,
isolated-import rules, API manifests, and source-distribution limits remain
unchanged. This packaging allowance does not promote MatPhys: only 3 of 10
opened source cases passed the frozen endpoint qualification, so no fresh
target evaluation was authorized and exact fallback remains the supported
behavior.

The native-continuum campaign adds 13 reviewed backend, qualification, and
source-value modules for JAX-FEM v2, MuJoCo Flex, and SOFA FEM v3. A clean
candidate build contains 595 wheel members and 2,425,511 compressed wheel
bytes; its source distribution contains 764 regular members and 2,322,777
compressed bytes. The wheel ratchets are therefore raised narrowly to 600
members and 2,440,000 bytes. Source-distribution limits remain unchanged. This
packaging adjustment does not promote any candidate: zero of three advanced to
source value, no source outcome partition or target was opened, and every
candidate retained the exact incumbent fallback.

The maintained five-backend support contract adds one stable Python module and
one installed, hash-bound JSON descriptor. A clean integrated build contains
604 wheel members and 2,468,567 compressed wheel bytes. The wheel ratchets are
therefore raised narrowly to 605 members and 2,475,000 bytes. The descriptor
records executable support and immutable evidence boundaries for DEFORM DLO,
MatPhys/Warp, JAX-FEM, MuJoCo Flex, and SOFA FEM without importing their
optional runtimes. This packaging adjustment does not change scientific
promotion: DEFORM remains scoped to its released DLO2 contract, while MatPhys
and all three native continuum candidates retain their registered negative
decisions and exact fallback.

The recursive corruption benchmark adds two NumPy-only installed modules: one
benchmark implementation and one grouped-CLI adapter. The immediately preceding
exact wheel artifact contains 606 members and 2,475,101 compressed bytes,
exposing a small pre-existing drift beyond the 605-member and 2,475,000-byte
ratchets. The exact integrated recursive candidate contains 609 wheel members
and 2,488,397 compressed bytes; it also contains the nonlinear-closure direct-
import module already integrated on current `main`. The wheel ratchets are
therefore set to 609 members and 2,550,000 bytes. Isolated-import, public-API,
console-script, and source-distribution limits remain unchanged. This allowance
supports controlled mechanism and local-closure infrastructure only; it does not
establish real-provider competence, physical transfer, covariance calibration,
intervention benefit, deployment safety, or state of the art.

The provider-neutral inference session adds two NumPy-only installed modules:
the orchestration implementation and its versioned v2 namespace. The wheel
member ratchet is therefore raised narrowly from 609 to 612 while the existing
2,550,000-byte limit is retained. The source distribution additionally includes
its exact API snapshot, bounded guide, and two focused self-test files; the
regular-member and compressed-size ratchets are raised to 1,370 and 3,275,000
bytes. This allowance creates no new estimator or scientific result. It only
makes the existing candidate/guard/exact-fallback separation accessible without
requiring a Prob4D-specific public signature.

The public-data covariance source barrier adds two private Deform360 package
modules without changing the public API or compressed-size ceilings. The wheel
member ratchet is therefore raised narrowly from 612 to 614, and the source
distribution regular-member ratchet from 1,370 to 1,372. This allowance only
packages the registered inventory and prediction-barrier implementation; it is
not source-gate, confirmation, calibration, or state-of-the-art evidence.

The query-conditional competence study adds four NumPy-only policy and protocol
modules without changing the exported public API, console command, optional
dependency surface, or compressed-size ceiling. The wheel member ratchet is
therefore raised narrowly from 614 to 618. These members define certificate,
cohort, and source-custody contracts; they do not package datasets, simulator
runtimes, physical outcomes, or a competence result.

The support-robust certificate work, its subsequent hardening, and the public
RCT real-decision study add six NumPy-only installed modules after the previous
619-member ratchet. A deterministic integrated build contains 625 wheel members
and 2,573,574 compressed wheel bytes. The wheel ratchets are therefore raised
narrowly to 625 members and 2,600,000 bytes. The RCT modules package the frozen
decision rule and protocol constants only; they do not package the public
dataset, source-test outcomes, confirmation outcomes, or a scientific result.

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
  tests/test_inference_v2.py \
  tests/test_inference_v2_public_api.py \
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
