# Three-repository installed-wheel golden path

This integration check exercises the released package boundary

```text
Prob4D -> Bayesian-PhysTwin -> Causal4D
```

without importing any repository directly from its source checkout.

## What the check proves

The runner builds one wheel from each repository, installs those wheels in a
fresh virtual environment, copies the test outside all source trees, clears
`PYTHONPATH`, and starts Python in isolated mode. The test then verifies:

1. Prob4D emits a deterministic, content-addressed strict causal-stream-v2
   observation with joint gauge covariance and complete source lineage.
2. Bayesian-PhysTwin independently reloads and validates that artifact,
   adapts it to the gauge-aware inference contract, and executes a
   deterministic update or its exact zero-update fallback.
3. Causal4D independently validates the installed Bayesian-PhysTwin provider,
   emits and round-trips a content-addressed `TwinBelief`, executes a
   counterfactual query through a deterministic `PhysTwinReplayProvider`
   implementation, and preserves staged posterior-mass accounting.
4. Future-dependent lineage, stream-version disagreement, missing metric
   calibration provenance, omitted anchor covariance, per-window gauge
   factors, and excessive covariance truncation are rejected fail-closed.

Validators remain implemented in their owning repositories. The integration
test shares only an immutable producer artifact and expected accept/reject
decisions.

## Local execution

From any directory with all three repositories checked out:

```bash
bash Bayesian-PhysTwin/scripts/run_three_repository_golden_path.sh \
  Bayesian-PhysTwin \
  Prob4D \
  Causal4D
```

The script needs a normal Python installation with `venv`, `pip`, `git`, and
network access for build/runtime dependencies. It removes all temporary build
and test environments when finished.

## GitHub Actions credential

`FlorianPfaff/Prob4D` is private. Configure a Bayesian-PhysTwin repository
secret named `PROB4D_READ_TOKEN` whose token has read-only contents access to
that repository. The workflow deliberately fails rather than silently
skipping the integration gate when this credential is absent.

Manual runs may select specific Prob4D and Causal4D refs. Pull-request and
scheduled runs use their `main` branches, so the weekly run also detects
downstream contract drift.
