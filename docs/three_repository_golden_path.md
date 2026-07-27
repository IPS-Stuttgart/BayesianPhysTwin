# Three-repository installed-wheel golden path

This integration gate executes the released package boundary

```text
Prob4D -> Bayesian-PhysTwin -> Causal4D
```

without importing any repository from its source checkout.

## What the gate proves

The runner requires three clean Git checkouts, records their exact commits,
creates immutable `git archive` source snapshots, builds one wheel from each
snapshot, and installs only those wheels in a fresh virtual environment. It
copies the integration test outside every source tree, clears `PYTHONPATH`, and
starts Python in isolated mode. The test then verifies:

1. Prob4D emits a deterministic, content-addressed strict causal-stream-v2
   observation with joint cross-window gauge covariance, complete metric-anchor
   provenance, and the exact tested Prob4D revision.
2. Bayesian-PhysTwin independently reloads and validates that artifact, adapts
   it to the gauge-aware inference contract, and executes a deterministic update
   or its exact zero-update fallback.
3. Causal4D independently reloads and validates the same observation archive,
   verifies that all source frames lie in `O-`, binds the observation lineage to
   a content-addressed `TwinBelief`, validates the installed Bayesian-PhysTwin
   provider, and executes a deterministic counterfactual query.
4. Posterior support reduction preserves staged probability-mass accounting and
   never requests replay for an exact-zero posterior cell.
5. A `RunManifestV2` binds all three repository commits, all three wheel hashes,
   package versions, the observation, `TwinBelief`, physical posterior, provider
   manifest, information boundary, protocol, split, baseline, method freeze, and
   claim ID. Promotion fails closed for dirty, incomplete, or tampered evidence.
6. Both independent consumers reject future-dependent lineage, future-payload
   access, stream-version disagreement, fixed-lag covariance falsely labelled as
   strict v2, changed metric-anchor source digests, missing calibration
   provenance, omitted anchor covariance, per-window gauge factors, duplicated
   gauge semantics, and excessive covariance truncation. Causal4D also rejects
   inconsistent composed posterior mass.

Validators remain implemented in their owning repositories. The integration
test shares only immutable artifacts and expected accept/reject decisions.

## Local execution

From any directory with clean checkouts of all three repositories:

```bash
bash Bayesian-PhysTwin/scripts/run_three_repository_golden_path.sh \
  Bayesian-PhysTwin \
  Prob4D \
  Causal4D
```

The script requires a normal Linux or macOS Python installation with `venv`,
`pip`, `git`, `tar`, and network access for build/runtime dependencies. It
removes all temporary source snapshots, wheels, and virtual environments on
exit.

## GitHub Actions credential

`FlorianPfaff/Prob4D` is private. Configure a Bayesian-PhysTwin repository secret
named `PROB4D_READ_TOKEN` whose token has read-only contents access to that
repository. Without the credential, the workflow emits an explicit warning and
records in the job summary that the cross-repository gate was not executed; it
does not admit or claim any three-repository evidence.

Manual runs may select specific Prob4D and Causal4D refs. Pull-request and
scheduled runs use their `main` branches, so a credentialed weekly run also
detects cross-repository contract drift.
