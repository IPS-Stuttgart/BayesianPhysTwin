# Three-Repository Compatibility Gates

The cross-repository CI has two deliberately different layers:

```text
checked-in producer-neutral fixture -> BayesianPhysTwin + Causal4D consumers
current Prob4D wheel -> BayesianPhysTwin wheel -> Causal4D wheel
```

The first layer always executes. The second layer is the evidence-bearing
installed-wheel golden path and requires read access to the private
`IPS-Stuttgart/Prob4D` repository.

## Always-Executed Consumer Fixture

Every relevant pull request checks out BayesianPhysTwin and the public
`IPS-Stuttgart/Causal4D` repository, installs both packages, and runs their
independent observation-contract, causal-lineage, and joint-gauge fixture tests.
This gate detects:

- schema or content-address drift;
- disagreement over strict causal-stream-v2 semantics;
- missing metric-anchor provenance;
- invalid joint cross-window gauge covariance claims;
- consumer-side lineage or artifact validation regressions; and
- accidental regression to repository names that predate the IPS-Stuttgart
  transfers.

The fixture gate does not execute the current Prob4D implementation. It is a
producer-neutral compatibility check and cannot by itself admit
three-repository evidence.

## Credentialed Installed-Wheel Golden Path

The evidence-bearing gate executes the released package boundary

```text
Prob4D -> BayesianPhysTwin -> Causal4D
```

without importing any repository from its source checkout. The runner requires
three clean Git checkouts, records their exact commits, creates immutable
`git archive` source snapshots, builds one wheel from each snapshot, and
installs only those wheels in a fresh virtual environment. It copies the
integration tests outside every source tree, clears `PYTHONPATH`, and starts
Python in isolated mode.

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

## Triggering and repository refs

The workflow runs for relevant BayesianPhysTwin pull requests and `main` changes,
weekly on schedule, manually, and for the repository-dispatch event types:

```text
prob4d-compatibility
causal4d-compatibility
```

A dispatch payload may provide `prob4d_ref` and `causal4d_ref`; absent values
resolve to `main`. Manual runs expose the same inputs. The resolved refs are
recorded in the job summary and included in the concurrency identity, so a run
for a specific producer commit is not silently replaced by an unrelated ref.

The receiver alone does not grant another repository permission to dispatch.
A producer-side workflow or operator must use a credential that can invoke
repository dispatch on `IPS-Stuttgart/BayesianPhysTwin`. Scheduled and manual
runs remain the fail-closed fallback when that organization credential is not
configured.

## Local Execution

From any directory with clean checkouts of all three repositories:

```bash
bash BayesianPhysTwin/scripts/run_three_repository_golden_path.sh \
  BayesianPhysTwin \
  Prob4D \
  Causal4D
```

The script requires a normal Linux or macOS Python installation with `venv`,
`pip`, `git`, `tar`, and network access for build/runtime dependencies. It
removes all temporary source snapshots, wheels, and virtual environments on
exit.

## GitHub Actions Credential Policy

`IPS-Stuttgart/Prob4D` is private. Configure a BayesianPhysTwin repository secret
named `PROB4D_READ_TOKEN` whose token has read-only contents access to that
repository.

Trusted same-repository pull requests, `main`, scheduled, manual, and
repository-dispatch runs must execute the credentialed installed-wheel path or
fail. This prevents a green trusted result from meaning that no producer
checkout, wheel, or test ran. External-fork pull requests cannot receive
repository secrets; they still run the producer-neutral consumer fixture and
receive an explicit private-gate-unavailable result that admits no current
Prob4D evidence.
