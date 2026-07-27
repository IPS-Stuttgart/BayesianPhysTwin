# Bayesian-PhysTwin

Reliability-aware Bayesian state and parameter estimation for PhysTwin-style
deformable digital twins.

Bayesian-PhysTwin treats learned tracks, masks, depth points, scene flow, and
point-cloud residuals as uncertain pseudo-measurements rather than
deterministic state. The package provides robust likelihoods, recursive
beliefs, explicit causal lineage, gauge-aware observation updates, guarded
fallback, and content-addressed evidence manifests.

## Project scope

```text
learned perception observations
+ spring-mass physical prior
+ Bayesian reliability and uncertainty
= guarded deformable-object state and parameter estimation
```

The reusable package covers:

- reliability-conditioned Gaussian/outlier and Student-t likelihoods;
- per-track reliability and drift-bias models;
- gauge-aware Prob4D observation contracts;
- prior-aware guarded state and parameter updates;
- low-rank graph-discrepancy beliefs;
- exact physical fallback when evidence is inadmissible;
- versioned Causal4D provider and belief artifacts; and
- reproducible run manifests binding repositories, runtime, inputs, and outputs.

This repository owns the Bayesian-PhysTwin package and its stable provider and
observation boundaries. It does not own Causal4D intervention semantics or
paper-level claim management.

## Installation

Bayesian-PhysTwin requires Python 3.10 or later.

```bash
python3 -m pip install -e .
```

For development:

```bash
python3 -m pip install -e ".[dev]"
```

Optional capabilities are installed explicitly:

```bash
python3 -m pip install -e ".[data]"       # remote ZIP data retrieval
python3 -m pip install -e ".[graph]"      # SciPy graph solvers
python3 -m pip install -e ".[vision]"     # OpenCV camera and cue workflows
python3 -m pip install -e ".[pyrecrest]"  # pinned downstream Bayesian filtering
```

A full local smoke run is available through:

```bash
bash scripts/local_smoke_test.sh
```

## Architecture

The reusable package is organized around four layers:

| Layer | Responsibility |
| --- | --- |
| Observation contracts | Validate causal timing, covariance, gauge factors, metric anchors, identity, and content hashes. |
| Bayesian inference | Convert observations into robust, reliability-aware state and parameter updates. |
| PhysTwin integration | Restore released physical artifacts, replay trajectories, and expose versioned provider operations. |
| Evidence and commands | Run stable workflows through the grouped CLI and bind outputs to reproducible manifests. |

Main source areas:

```text
src/bayesian_phystwin/
    observation_belief.py              versioned observation artifact
    observation_belief_gauge_adapter.py
    gauge_aware_belief.py              joint state/nuisance update
    phystwin_belief.py                 recursive guarded belief
    causal4d_provider_v1.py            versioned downstream provider
    run_manifest_v2.py                 content-addressed run provenance
    cli/                               grouped and compatibility commands
```

## Stable command surface

`bpt` is the supported command namespace. The 79 historical `bpt-*` console
scripts remain installed only as frozen compatibility aliases for published
scripts and result manifests. New commands are grouped-only registry entries
and must not add another top-level executable.

```bash
bpt --help
bpt commands
bpt commands --all
bpt experiment list
bpt experiment run <id> --help
bpt diagnostic list
bpt archived list
```

The default help and command listing expose stable interfaces and current
experiments only. Diagnostics and archived commands remain directly queryable
without being promoted in the primary workflow.

Stable workflows:

| Command | Purpose |
| --- | --- |
| `bpt provider manifest` | Emit the versioned Causal4D provider capability manifest. |
| `bpt observation validate` | Validate or summarize an `ObservationBeliefV1` artifact. |
| `bpt residual replay` | Replay exported residuals through the robust likelihood. |
| `bpt benchmark synthetic` | Run the controlled fixed-graph benchmark. |
| `bpt run manifest` | Create or validate content-addressed run provenance. |
| `bpt evidence summarize` | Summarize matched guarded prospective evidence. |

The declarative registry records each grouped route, lifecycle status,
implementation target, optional dependency extras, owning milestone, and frozen
compatibility alias. See [Command surface](docs/command_surface.md).

## Minimal examples

Replay an exported residual table:

```bash
bpt residual replay examples/residuals_demo.csv \
  --summary-json outputs/residuals_demo/summary.json \
  --scored-csv outputs/residuals_demo/scored.csv
```

Run the controlled synthetic benchmark:

```bash
bpt benchmark synthetic \
  --seeds 1000:1020 \
  --conditions clean,iid,correlated \
  --action-modes dynamic,quasi_static \
  --output-json runs/synthetic_v3/results.json \
  --output-csv runs/synthetic_v3/aggregate.csv \
  --output-reliability-csv runs/synthetic_v3/reliability.csv
```

Create a reproducible run manifest after outputs are immutable:

```bash
bpt run manifest create runs/example/manifest.json \
  --run-id phystwin-example-v1 \
  --classification confirmatory \
  --statistical-unit interaction \
  --repository-root . \
  --artifact-root runs/example \
  --output-artifact metrics=metrics.json
```

## Cross-repository interfaces

[Prob4D](https://github.com/FlorianPfaff/Prob4D) produces versioned
probabilistic observations with causal lineage, retained gauge uncertainty, and
metric-anchor provenance. Bayesian-PhysTwin independently validates those
artifacts before any state or parameter update.

[Causal4D](https://github.com/FlorianPfaff/Causal4D) consumes versioned
Bayesian-PhysTwin belief and replay-provider artifacts. Causal abduction,
intervention semantics, language-conditioned evidence, and physical acquisition
protocols remain owned by Causal4D.

[BayesianPhysTwin-Paper](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper)
is the canonical source for current project status, claim language, figures,
and paper-facing evidence. The historical
`2026-07-Causal4D-BPT-Paper` repository is not an operational source of truth.

## Documentation map

- [Experiment and evidence index](docs/experiment_index.md)
- [Command surface and compatibility policy](docs/command_surface.md)
- [Observation-belief contract](docs/observation_belief_contract.md)
- [Gauge-aware observation update](docs/gauge_aware_observation_update.md)
- [Prior-aware guarded update](docs/prior_aware_guarded_update.md)
- [Causal4D provider API v1](docs/causal4d_provider_v1.md)
- [Reproducible run manifests](docs/reproducible_runs.md)
- [Compute and remote-run policy](docs/compute.md)
- [Contributing guide](CONTRIBUTING.md)
- [Support and security](SUPPORT.md)

The experiment index links frozen positive and negative results without turning
the root README into a live laboratory notebook.

## Development

```bash
python3 -m pytest
python3 -m ruff check src tests
python3 -m mypy src/bayesian_phystwin/run_manifest.py \
  src/bayesian_phystwin/repository_provenance.py \
  src/bayesian_phystwin/run_manifest_v2.py \
  src/bayesian_phystwin/cli/main.py \
  src/bayesian_phystwin/cli/run_manifest.py
```

Large datasets, checkpoints, rendered videos, and raw runs belong outside git
under ignored paths such as `data/`, `checkpoints/`, `runs/`, and `outputs/`.
