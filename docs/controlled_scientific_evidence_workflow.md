# Controlled scientific evidence workflow

## Purpose

The permanent `Paper evidence contracts` workflow has an opt-in manual execution
path for three registered target-free studies. It produces numerical result
artifacts rather than treating unit tests or a green workflow as scientific
evidence:

- `simulation-based-calibration`: the frozen 512-replicate exact-model versus
  correlated-misspecification calibration study;
- `synthetic-benchmark-sbc`: the 512-replicate end-to-end finite-grid posterior
  study with matched, underdispersed, and overdispersed likelihood arms; and
- `recursive-corruption`: the fixed 50-seed corruption benchmark together with
  the preregistered innovation-threshold selectivity diagnostic.

`contracts-only` remains the default manual mode. Pull requests and ordinary
pushes continue to run only the evidence-contract validation job. The numerical
study job is admitted only for `workflow_dispatch` on `main` after the contract
job passes.

## Dispatch

Run one study from the GitHub Actions interface or with the GitHub CLI:

```bash
gh workflow run paper-evidence.yml \
  --repo IPS-Stuttgart/BayesianPhysTwin \
  --ref main \
  -f evidence_study=simulation-based-calibration \
  -f verify_replay=true
```

The valid `evidence_study` values are:

```text
contracts-only
simulation-based-calibration
synthetic-benchmark-sbc
recursive-corruption
all-controlled
```

`all-controlled` executes all three registered studies sequentially. It does not
combine their scientific decisions. A failed or negative study cannot be rescued
by another study.

When `verify_replay=true`, every selected study is executed a second time in the
same frozen environment. The claim-bearing JSON and CSV outputs must be
byte-identical. Replay is an implementation-reproducibility gate, not independent
scientific replication.

## Exact execution surface

The workflow checks out the exact reviewed `main` revision, builds one wheel, and
imports the stable `bayesian_phystwin` package from that installed wheel. The
research-only `bayesian_phystwin_experiments` namespace is intentionally excluded
from the public wheel, so the workflow copies only that namespace into an
isolated source overlay from the same exact Git revision. It verifies that:

- `bayesian_phystwin` resolves outside the checkout's `src/` tree; and
- `bayesian_phystwin_experiments` resolves inside the isolated exact-revision
  research overlay.

The bundle records the wheel SHA-256, installed packages, Python and runner
identity, repository status, and SHA-256 values for the study runners, protocol,
research modules, and orchestration code.

The versioned orchestration entry point is:

```text
scripts/science/run_controlled_evidence_workflow_v1.py
```

It invokes the registered study runners without changing their estimator,
protocol, seeds, conditions, thresholds, or decision rules.

## Evidence gates

The orchestration layer rejects malformed, duplicate-key, or non-finite JSON and
applies study-specific checks:

- simulation-based calibration must return the registered exact-model and fixed
  misspecification decision;
- synthetic benchmark SBC must place the matched posterior ahead of both fixed
  dispersion controls on mean KS distance and nominal-90% coverage error; and
- recursive corruption must retain all 50 seeds, seven conditions, five methods,
  1,750 sequence-method records, 300 corrupted sequences per method, the exact
  seven-threshold diagnostic grid, `selection_authorized=false`, and zero exact
  fallback violations.

A valid negative scientific decision is retained in `decision.json` and the
bundle summary. The workflow exits nonzero only after the artifact publication
step has had an opportunity to retain the result and failure receipts.

## Retained artifact

Each run uploads:

```text
controlled-scientific-evidence-<study>-<run-id>
```

for 90 days. The artifact contains:

- the exact command and exit status for every primary and replay execution;
- complete result JSON and long-form CSV where applicable;
- stdout/stderr logs;
- one per-study `decision.json`;
- `bundle-summary.json`;
- environment, source, and wheel identities; and
- `manifest.json`, whose `manifest_id` content-addresses the complete retained
  file roster.

The workflow does not commit generated outcomes automatically. Paper promotion
still requires a reviewed evidence intake that binds the Actions run, artifact,
manifest ID, exact source revision, interpretation, and claim boundary.

## Scientific boundary

All three modes are controlled target-free evidence. They open no Deform360
confirmation object, DLO4/DLO5 or held-v8 payload, Prob4D target outcome, real
provider result, or Causal4D physical execution. They do not establish real-data
calibration, unseen-object transfer, intervention benefit, deployment safety, or
state of the art.
