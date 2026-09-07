# Residual-model research code

The owner-requested 2026-09-07 integration retains twelve GP/DP research PRs.
Start with [Gaussian processes](gaussian-processes.md) or
[Dirichlet processes](dirichlet-processes.md). The installed package, default
predictor, frozen numerical protocols, and existing evidence are not changed.

## Code and evidence ownership

Executable models remain in the repository's existing research namespaces:
`experiments/`, `src/bayesian_phystwin_experiments/`, and `scripts/remote/` or
`scripts/experiments/`. None is a new stable `bpt` command or installed public API.
Root-level unit tests use optional-dependency skips where appropriate; the
maintained CPU-only residual test lane installs the dependencies and requires
**zero skips** for every imported suite.

[The import catalog](imports-2026-09-07.json) records every source PR/head,
original file/blob/hash, current source path, and deliberately excluded outcome
or documentation path. Original numerical evidence continues to refer to its
executed source revision, not the mechanically maintained integration snapshot.
Source formatting/import hygiene and the PR #962 namespace repair are recorded
separately. No result hash is rewritten to pretend that newer code was executed.

The two independently developed `dp_residual_development_v1` implementations
are **not duplicates**. PR #960 retains that namespace; PR #962 uses
`experiments/dp_residual_kinematic_v1/`. Its imports are changed consistently,
while numerical statements and constants remain the same.

Paper-facing reports belong in the [paper experiment notebook](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/tree/main/experiments).
Existing positive controls, negative results, failed continuation gates,
historical-selection limitations, and separate failed execution routes remain
there or in the exact original PR. Outcome-bearing JSON and competing result
reports are not copied into the public active experiment tree. Source PR ancestry
and the catalog retain access to every omitted original blob.

## CPU-only implementation checks

In an isolated Python 3.12 environment, install the development package and the
optional test dependencies, then run:

```bash
python -m pip install -e '.[dev]' numpy==1.26.4 scipy==1.15.3 scikit-learn==1.7.2
python scripts/ci/check_residual_experiments.py
```

The runner starts each historical suite in a separate process to avoid collisions
between generic local `model` and `run` module names. It checks only constructed
numerical fixtures and temporary test files. It does not download datasets,
restore simulator checkpoints, fit on recordings, or score target cohorts.

Run a single reviewed suite with `--pr 952`. JSON receipts can be written with
`--output-dir /tmp/residual-checks`. No command here authorizes a scientific rerun.

## Execution workflows and frozen reproduction

All newly proposed one-shot workflows and trigger requests are retained as
**inactive byte-exact history** under
[`archive/research-workflows/residual-models-2026-09-07/`](../../archive/research-workflows/residual-models-2026-09-07/).
They are outside `.github/workflows/` and cannot trigger on merging. The existing
changed-source preflight owns the maintained CPU-only test job; this integration
adds **no active workflow files** and requests no write permissions.

Data-bearing scripts keep their original information boundaries. Historical
reproduction uses the original executed revision, owning protocol, and declared
cache/runtime; shell launchers and source-bound run identities in this archive
are not automatic authorization to rerun terminal or protected studies.

## Deferred work

PRs #944 and #947 are earlier overlapping preparations without the corresponding
focused regression suites. PR #964 is a distinct prospective temporal-discrepancy
extension whose native-data adapter is not validated by this consolidation.
They remain separate reviews. This merge does not merge unrelated active-sensing,
causal-attribution, or other-repository feature branches.
