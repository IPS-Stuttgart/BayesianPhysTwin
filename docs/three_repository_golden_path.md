# Three-Repository Compatibility Gates

The cross-repository CI has three deliberately different blocking layers plus
an advisory latest-main canary:

```text
checked-in producer-neutral fixture -> BayesianPhysTwin + Causal4D consumers
locked Prob4D wheel -> current BayesianPhysTwin wheel -> locked Causal4D wheel
locked Prob4D wheel -> locked BayesianPhysTwin wheel -> locked Causal4D wheel
latest Prob4D main -> current BayesianPhysTwin wheel -> latest Causal4D main (canary)
```

The blocking layers execute on every relevant pull request. They use the
committed [ecosystem compatibility lock](ecosystem_compatibility.md) unless an
explicit manual or repository-dispatch override selects companion refs. The
canary runs on the weekly schedule or when explicitly requested and is allowed
to fail without invalidating a compatible BayesianPhysTwin change.

All layers use the public canonical `IPS-Stuttgart/Prob4D` and
`IPS-Stuttgart/Causal4D` repositories without repository secrets.

## Always-Executed Consumer Fixture

Every relevant pull request checks out BayesianPhysTwin and the locked public
`IPS-Stuttgart/Causal4D` revision, installs both packages, and runs their
independent observation-contract, causal-lineage, joint-gauge, command-registry,
and compatibility-lock fixture tests. This gate detects:

- schema or content-address drift;
- disagreement over strict causal-stream-v2 semantics;
- missing metric-anchor provenance;
- invalid joint cross-window gauge covariance claims;
- consumer-side lineage or artifact validation regressions; and
- drift in the workflow's transfer-safe repository, lock, and dispatch policy.

The fixture gate does not execute Prob4D. It is a producer-neutral compatibility
check and cannot by itself admit three-repository evidence. It also does not
replace either installed-wheel path.

## Current-Source Installed-Wheel Golden Path

The primary forward-compatibility gate executes the package boundary

```text
Prob4D -> current BayesianPhysTwin -> Causal4D
```

without importing any repository from its source checkout. The runner requires
three clean public Git checkouts, records their exact commits, creates immutable
`git archive` source snapshots, builds one wheel from each snapshot, and
installs only those wheels in a fresh virtual environment. It copies the
integration tests outside every source tree, clears `PYTHONPATH`, and starts
Python in isolated mode.

Before the integration tests, the installed `bpt ecosystem validate` command
requires all three packages. For ordinary pull requests and pushes it also
requires the exact locked Prob4D and Causal4D commits and exact locked package
versions. The resulting content-addressed report is uploaded with the workflow
artifacts. Explicit ref overrides are reported and validated against compatible
package lines, but are not presented as exact-lock reproduction.

The test then verifies:

1. Prob4D emits a deterministic, content-addressed strict causal-stream-v2
   observation with joint cross-window gauge covariance, complete metric-anchor
   provenance, and the exact tested Prob4D revision.
2. The installed Prob4D wheel publishes stable project identity
   `github-repository-id:1295794737`, canonical repository
   `IPS-Stuttgart/Prob4D`, and the historical artifact alias
   `FlorianPfaff/Prob4D`. BayesianPhysTwin continues to require the historical
   alias inside frozen observation schemas instead of rewriting existing
   content-addressed artifacts.
3. BayesianPhysTwin independently reloads and validates that artifact, adapts
   it to the gauge-aware inference contract, and executes a deterministic update
   or its exact zero-update fallback.
4. Causal4D independently reloads and validates the same observation archive,
   verifies that all source frames lie in `O-`, binds the observation lineage to
   a content-addressed `TwinBelief`, validates the installed BayesianPhysTwin
   provider, and executes a deterministic counterfactual query.
5. Posterior support reduction preserves staged probability-mass accounting and
   never requests replay for an exact-zero posterior cell.
6. A `RunManifestV2` binds all three repository commits, all three wheel hashes,
   package versions, the observation, `TwinBelief`, physical posterior, provider
   manifest, information boundary, protocol, split, baseline, method freeze, and
   claim ID. Promotion fails closed for dirty, incomplete, or tampered evidence.
7. Both independent consumers reject future-dependent lineage, future-payload
   access, stream-version disagreement, fixed-lag covariance falsely labelled as
   strict v2, changed metric-anchor source digests, missing calibration
   provenance, omitted anchor covariance, per-window gauge factors, duplicated
   gauge semantics, and excessive covariance truncation. Causal4D also rejects
   inconsistent composed posterior mass.

Validators remain implemented in their owning repositories. The integration
test shares only immutable artifacts and expected accept/reject decisions.

## Exact Historical Lock Reproduction

A separate blocking job reads the BayesianPhysTwin, Prob4D, and Causal4D commits
from the same packaged lock, checks out all three exact revisions, verifies each
checkout identity, and invokes the golden-path script from the historical
BayesianPhysTwin source itself. This distinguishes two questions that must not be
conflated:

- does the current BayesianPhysTwin change remain compatible with the locked
  companions; and
- does the lock still identify the exact source combination whose successful run
  created the compatibility evidence?

The historical lane runs only when the committed lock is in force. An explicit
companion override is a diagnostic compatibility run, not a reproduction of the
version-1 lock.

## Latest-Main Canary

The latest-main canary repeats the complete installed-wheel route using current
Prob4D and Causal4D `main`. It validates package compatibility and produces its
own report, but does not claim exact lock reproduction. Because it is a
job-level `continue-on-error` lane, upstream development drift is visible without
making an otherwise locked-compatible BayesianPhysTwin pull request randomly
red. A canary failure should be repaired in the owning repository and followed
by a reviewed lock update after the locked route passes.

## Transfer-Safe Repository Identity

The workflow checks out the public canonical repository
`IPS-Stuttgart/Prob4D` directly. The historical string `FlorianPfaff/Prob4D` is
retained only where it is part of a frozen content-addressed observation or
provider contract. Repository navigation and artifact semantics are therefore
kept separate.

The installed wheel must independently report:

```text
project_id = github-repository-id:1295794737
canonical_repository = IPS-Stuttgart/Prob4D
frozen_artifact_repository = FlorianPfaff/Prob4D
```

A repository transfer must not rewrite an existing artifact, provider manifest,
run manifest, or content identifier. A future artifact schema can bind the stable
project ID explicitly; current frozen schemas continue to require their exact
historical source string.

Because all three repositories are public, missing checkout access is a real
compatibility failure rather than a condition for skipping the golden path.
Pull requests from forks receive the same read-only public integration coverage
as organization branches.

## Triggering and Repository Refs

The workflow runs for relevant BayesianPhysTwin pull requests and `main` changes,
weekly on schedule, manually, and for the repository-dispatch event types:

```text
prob4d-compatibility
causal4d-compatibility
```

Absent ref overrides resolve to the exact commits in
`ecosystem_compatibility_v1.json`, not moving branches. A dispatch payload may
provide `prob4d_ref` and `causal4d_ref`; manual runs expose the same optional
inputs. The selected and locked refs are recorded in the job summary and the
workflow reports whether the exact lock was enforced.

The receiver alone does not grant another repository permission to dispatch.
A producer-side workflow or operator must use a credential that can invoke
repository dispatch on `IPS-Stuttgart/BayesianPhysTwin`; the integration job
itself requires only the normal read-only `GITHUB_TOKEN` and public checkout
access.

## Local Execution

From any directory with clean checkouts of all three repositories:

```bash
bash BayesianPhysTwin/scripts/run_three_repository_golden_path.sh \
  BayesianPhysTwin \
  Prob4D \
  Causal4D
```

The current script always validates that all three installed wheels lie on
compatible package lines. To persist the report and require exact locked
companion commits:

```bash
THREE_REPOSITORY_COMPATIBILITY_REPORT="$PWD/compatibility.json" \
THREE_REPOSITORY_REQUIRE_LOCKED_REVISIONS=true \
  bash BayesianPhysTwin/scripts/run_three_repository_golden_path.sh \
    BayesianPhysTwin Prob4D Causal4D
```

To reproduce the historical lock locally, check out all three revisions from the
lock and run the script from that locked BayesianPhysTwin checkout. The script
requires a normal Linux or macOS Python installation with `venv`, `pip`, `git`,
`tar`, and network access for build/runtime dependencies. It removes temporary
source snapshots, wheels, and virtual environments on exit; an explicitly
requested compatibility report remains outside that temporary root.

## GitHub Actions Public-Repository Policy

The workflow grants only `contents: read`, pins external actions by immutable
commit, disables persisted checkout credentials, and checks out all three public
repositories directly. It does not probe repository access with a personal
token and does not accept a `PROB4D_READ_TOKEN` secret.

Checkout, build, artifact, lock-validation, or compatibility failure fails the
blocking jobs explicitly. The producer-neutral fixture is never relabelled as a
full three-repository result, the historical lane is not confused with current
forward compatibility, and the latest-main canary is clearly separated from
exact-lock evidence.
