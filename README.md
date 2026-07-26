# Bayesian PhysTwin

Reliability-aware Bayesian state and parameter estimation for PhysTwin-style
deformable digital twins.

Bayesian-PhysTwin treats lifted tracks, masks, depth points, scene flow, and
point-cloud residuals as uncertain pseudo-measurements. It adds explicit
reliability, robust likelihoods, posterior utilities, causal artifact contracts,
and guarded fallback around a physical simulator rather than replacing the
simulator with an unconstrained predictor.

## Scope and repository boundaries

This repository owns:

- Bayesian observation, belief, reliability, and discrepancy models;
- versioned observation, provider, replay, and run-manifest contracts;
- reusable PhysTwin artifact and evaluation utilities;
- Bayesian-PhysTwin experiments and diagnostics.

Related repositories have deliberately separate ownership:

- [Prob4D](https://github.com/FlorianPfaff/Prob4D) produces versioned learned
  observation artifacts and their gauge uncertainty;
- [Causal4D](https://github.com/FlorianPfaff/Causal4D) consumes Bayesian-PhysTwin
  beliefs and replay capabilities for abduction, intervention, and
  counterfactual prediction;
- [BayesianPhysTwin-Paper](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper)
  tracks current project notes, evidence status, claims, figures, and paper
  artifacts.

Operational Causal4D milestones and paper-facing interpretations are maintained
in those repositories, not duplicated here. See
[the integration contract](docs/phystwin_integration.md) and
[the Causal4D migration note](docs/causal4d_migration.md).

## Current evidence

On the official ordered 22-case PhysTwin cohort, the frozen Bayesian anchor
improves equal-case Chamfer distance by 12.09% and track error by 12.78% over
re-evaluated released `inference.pkl` trajectories. This is an improvement over
released PhysTwin, not an overall state-of-the-art claim against later methods.
The frozen protocol, uncertainty intervals, render metrics, provenance, and
ownership boundary are recorded in
[the full-22 evidence report](docs/phystwin_sota_22_v1.md).

## Installation

```bash
python3 -m pip install -e ".[dev,data,graph]"
bash scripts/local_smoke_test.sh
```

Optional extras are intentionally separated:

| Extra | Purpose |
|---|---|
| `data` | selective release-archive retrieval |
| `graph` | SciPy-backed graph and sparse inference utilities |
| `vision` | raw-camera and correspondence processing |
| `pyrecest` | optional source-only recursive-estimation diagnostics |

## Stable command surface

The supported entry point is the grouped `bpt` command:

```bash
bpt --help
bpt provider manifest --help
bpt observation validate --help
bpt run manifest --help
bpt residual replay --help
bpt benchmark synthetic --help
```

These routes are stable interfaces. The package imports their implementations
lazily, so listing commands does not require graph, vision, or experiment-only
dependencies.

## Experiments, diagnostics, and archived commands

Research commands are recorded in a typed registry with their grouped route,
legacy alias, lifecycle status, optional dependencies, and owning protocol or
milestone.

```bash
# Current experiment commands
bpt experiment list
bpt experiment describe confirm-phystwin-bayesian-anchor
bpt experiment run confirm-phystwin-bayesian-anchor --help

# Non-promotable audits and analyses
bpt diagnostic list
bpt diagnostic describe audit-phystwin-calibration

# Frozen historical paths, omitted from current experiment listings
bpt archive list

# Machine-readable complete registry
bpt commands list --json
bpt commands describe bpt-confirm-phystwin-bayesian-anchor --json
```

The current paper-facing experiment routes include the official PhysTwin
full-cohort evaluation and the sealed Deform360 bias-aware protocol. Run
`bpt experiment list` for the authoritative current set rather than copying a
static command inventory from this README.

The installed `bpt-*` executables remain available for frozen environments and
historical reproduction. They are a compatibility surface, not the extension
mechanism for new work. New commands must be added to the registry and invoked
through `bpt`; no new top-level console script should be added.

See [Command surface and compatibility policy](docs/command_surface.md) for the
status definitions, migration rules, and contribution procedure.

## Common stable workflows

Replay an exported residual table through the robust likelihood:

```bash
bpt residual replay examples/residuals_demo.csv \
  --summary-json outputs/residuals_demo/summary.json \
  --scored-csv outputs/residuals_demo/scored.csv
```

See [Residual replay](docs/residual_replay.md) for the canonical export schema,
statistical model, and output metrics.

Run the controlled fixed-graph benchmark used for parameter recovery,
calibration, correlated corruption, and action-informativeness ablations:

```bash
bpt benchmark synthetic \
  --seeds 1000:1020 \
  --conditions clean,iid,correlated \
  --action-modes dynamic,quasi_static \
  --output-json runs/synthetic_v3/results.json \
  --output-csv runs/synthetic_v3/aggregate.csv
```

See [Synthetic benchmark](docs/synthetic_benchmark.md) for the complete protocol
and baseline definitions.

## Documentation

- [PhysTwin integration and artifact boundary](docs/phystwin_integration.md)
- [Advanced inference protocols](docs/phystwin_advanced_inference.md)
- [Full-22 evidence report](docs/phystwin_sota_22_v1.md)
- [Command surface and compatibility policy](docs/command_surface.md)
- [Compute conventions](docs/compute.md)

Experiment-specific reports remain in `docs/` and versioned result directories.
The root README is intentionally limited to stable interfaces, current entry
points, and repository boundaries.

## Repository layout

```text
src/bayesian_phystwin/   reusable package and versioned contracts
tests/                   unit, contract, packaging, and CLI tests
examples/                small synthetic demos
configs/compute/         host-specific run defaults
scripts/remote/          remote experiment helpers
docs/                    protocols, evidence reports, and integration notes
results/                 compact versioned result artifacts
```

Large datasets, checkpoints, rendered videos, and raw runs should remain outside
git under ignored local paths such as `data/`, `runs/`, `outputs/`, and
`checkpoints/`.

## Compute

GPU experiments are intended for the configured `gpuserver6000` and
`gpuserver4090` hosts through the project jumpserver. See
[Compute conventions](docs/compute.md) for current run and provenance rules.
