# Bayesian PhysTwin

Reliability-aware Bayesian belief estimation for PhysTwin-style deformable
digital twins.

Bayesian-PhysTwin treats learned tracks, depth points, masks, flow, and related
4-D perception outputs as noisy pseudo-measurements. It combines them with a
PhysTwin physical prior while keeping observation reliability, structured
covariance, physical-parameter uncertainty, and simulator discrepancy
explicit. When an update is not identifiable or fails a prospective guard, the
library retains the physical baseline instead of silently applying an unsafe
correction.

## Scientific scope

This repository owns:

- versioned observation, physical-linearization, belief, provider, and run
  provenance contracts;
- reliability-aware and structured robust likelihoods;
- recursive, gauge-aware, and prior-aware Bayesian updates;
- guarded predictive-discrepancy and fallback logic;
- adapters and evaluation utilities for official PhysTwin artifacts.

A predictive readout-discrepancy belief is not automatically a corrected latent
physical state. Released trajectories also do not identify a unique physical
cause. Experiments and papers should preserve that distinction explicitly.

## Current evidence

On the official ordered 22-case PhysTwin cohort, the frozen Bayesian anchor
improves equal-case Chamfer distance by **12.09%** and track error by **12.78%**
relative to re-evaluated released `inference.pkl` trajectories. The result is
better than released PhysTwin under that protocol, but it is not overall state
of the art against later published methods. A simple last-residual comparator
is also marginally better on deterministic track error, and raw posterior
covariance is not calibrated.

See the [full-22 evidence report](docs/phystwin_sota_22_v1.md) for the frozen
cohort, uncertainty intervals, render metrics, provenance, and permitted claim
boundary.

## Architecture

```text
Prob4D or another 4-D perception feeder
                │
                ▼
       ObservationBeliefV1 ───────────────┐
                                          │
       PhysTwin physical prior ───────────┼──► robust likelihood
                                          │    + guarded Bayesian update
                                          │
                                          └──► predictive belief
                                               or exact fallback
                                                        │
                                                        ▼
                                           Causal4D provider artifacts
```

[Prob4D](https://github.com/FlorianPfaff/Prob4D) can export the portable
`ObservationBeliefV1` contract. Bayesian-PhysTwin owns the reliability-aware
belief update and PhysTwin provider boundary. [Causal4D](https://github.com/FlorianPfaff/Causal4D)
separately owns abduction, intervention, and counterfactual prediction.

## Installation

```bash
python3 -m pip install -e ".[dev,data,graph]"
bash scripts/local_smoke_test.sh
bpt --help
```

The base package requires only NumPy. Optional dependency groups add development
tools, selective data retrieval, sparse graph routines, vision utilities, or
the pinned PyRecEst integration.

## Stable command surface

The grouped `bpt` interface is the stable entry point. Historical `bpt-*`
commands remain available for compatibility with frozen experiments.

| Command | Purpose |
| --- | --- |
| `bpt provider manifest` | Print the versioned Causal4D provider capability manifest. |
| `bpt observation validate` | Validate or summarize an `ObservationBeliefV1` artifact. |
| `bpt residual replay` | Replay exported residuals through the robust likelihood. |
| `bpt benchmark synthetic` | Run the controlled synthetic benchmark. |
| `bpt run manifest` | Create or validate content-addressed run provenance. |

Replay the bundled residual example:

```bash
bpt residual replay examples/residuals_demo.csv \
  --summary-json outputs/residuals_demo/summary.json \
  --scored-csv outputs/residuals_demo/scored.csv
```

See [residual replay](docs/residual_replay.md) for the export schema,
statistical model, and output metrics.

## Reproduce the controlled benchmark

The following command runs the documented fixed-graph benchmark used for
parameter recovery, calibration, corruption, and action-informativeness
controls:

```bash
bpt benchmark synthetic \
  --seeds 1000:1020 \
  --conditions clean,iid,correlated \
  --action-modes dynamic,quasi_static \
  --bias-process-variance 1e-5 \
  --bias-initial-variance 1e-7 \
  --bias-cue-persistence 0.85 \
  --bias-cue-threshold 0.20 \
  --bias-minimum-run-length 5 \
  --output-json runs/synthetic_v3/results.json \
  --output-csv runs/synthetic_v3/aggregate.csv \
  --output-reliability-csv runs/synthetic_v3/reliability.csv
```

The complete frozen protocol and baseline definitions are in the
[synthetic benchmark documentation](docs/synthetic_benchmark.md).

## Python API

Loading an observation belief revalidates its schema, content address,
covariance, identities, and exclusive causal cutoff:

```python
from bayesian_phystwin import load_observation_belief

belief = load_observation_belief("observation_belief.npz")
print(belief.summary())
```

The portable contract is documented in
[ObservationBeliefV1](docs/observation_belief_contract.md). The
[gauge-aware adapter](docs/gauge_aware_observation_update.md) and
[prior-aware guarded update](docs/prior_aware_guarded_update.md) describe the
state-update and exact-fallback boundaries.

## Documentation map

- [Experiment and evidence index](docs/experiment_index.md): frozen reports,
  negative results, experimental command families, and placement policy.
- [Causal4D provider v1](docs/causal4d_provider_v1.md): supported provider
  surface and provenance boundary.
- [PhysTwin integration](docs/phystwin_integration.md): upstream artifacts,
  residual export, cue sidecars, and likelihood ownership.
- [Compute conventions](docs/compute.md): GPU-host and remote-run policy.
- [Canonical project notes](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper):
  current scope, evidence status, figures, result artifacts, and paper claims.

## Repository layout

```text
src/bayesian_phystwin/   reusable Python package and versioned contracts
tests/                   unit, conformance, and integration tests
examples/                small synthetic inputs and demos
configs/                 frozen experiment and compute configurations
scripts/                 local and remote execution helpers
docs/                    contracts, protocols, evidence, and experiment index
results/                 compact frozen evidence and audit artifacts
```

Large datasets, checkpoints, rendered videos, and raw runs should remain outside
git in ignored paths such as `data/`, `checkpoints/`, `runs/`, and `outputs/`.

New experiment narratives and status updates should be added to a dedicated
protocol or evidence document and linked from the experiment index, rather than
expanding the root README.