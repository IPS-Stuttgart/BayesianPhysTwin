# Bayesian PhysTwin

Reliability-aware Bayesian state and parameter estimation for PhysTwin-style
deformable digital twins.

The first target is a lightweight layer around PhysTwin outputs: lifted point
tracks, masks, depth points, scene flow, and point-cloud residuals are treated
as noisy pseudo-measurements with explicit reliability. The longer-term goal is
posterior inference over deformable state, material/contact parameters, and
possibly spring-graph topology.

## Research Direction

PhysTwin estimates a physical digital twin from sparse video. This repository
adds the estimation layer needed for robust robotics:

```text
learned perception observations
+ spring-mass physical prior
+ Bayesian reliability / uncertainty layer
= robust deformable-object state and parameter estimation
```

Initial scope:

- reliability-weighted pseudo-measurements for tracks, depth, masks, and flow
- reliability-conditioned Gaussian/outlier mixture likelihoods
- per-track Markov reliability and robust random-walk drift bias
- robust residual losses for inverse-physics fitting
- ensemble/posterior utilities for low-dimensional physical parameters
- exact and hierarchical multi-interaction parameter pooling
- regularized spatial spring regions and explicit damping sweeps
- causal, action-conditioned low-rank simulator discrepancy
- raw camera/mask cue recovery and paired moving-block evaluation
- reproducible experiment configs and remote-run scripts

## Repository Layout

```text
src/bayesian_phystwin/   reusable Python package
tests/                   unit tests for estimation utilities
examples/                small synthetic demos
configs/compute/         host-specific run defaults
scripts/remote/          GPU-server helpers
docs/                    notes on compute and integration
```

Large datasets, checkpoints, rendered videos, and raw runs should stay out of
git. Use `runs/`, `outputs/`, `checkpoints/`, and `data/` locally or on the GPU
servers; these paths are ignored by default.

## Quick Start

```bash
python3 -m pip install -e ".[dev]"
bash scripts/local_smoke_test.sh
```

Replay an exported residual table through the robust likelihood:

```bash
bpt-replay-residuals examples/residuals_demo.csv \
  --summary-json outputs/residuals_demo/summary.json \
  --scored-csv outputs/residuals_demo/scored.csv
```

See [docs/residual_replay.md](docs/residual_replay.md) for the canonical export
schema, statistical model, and output metrics.

Run the controlled fixed-graph benchmark used for parameter recovery,
calibration, correlated corruption, and action-informativeness ablations:

```bash
bpt-synthetic-benchmark \
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

See [docs/synthetic_benchmark.md](docs/synthetic_benchmark.md) for the complete
protocol and baseline definitions.

Export the exact tracked-point residuals from an official PhysTwin case and
immediately replay them through the reliability model:

```bash
bpt-export-phystwin-residuals \
  data/different_types/CASE/final_data.pkl \
  experiments/CASE/inference.pkl \
  runs/CASE/residuals.csv \
  --replay-summary-json runs/CASE/replay.json \
  --scored-csv runs/CASE/scored.csv
```

Generate a continuous neighbor-motion cue sidecar before replay when only
PhysTwin's processed boolean validity mask is available:

```bash
bpt-build-phystwin-cues \
  data/different_types/CASE/final_data.pkl \
  runs/CASE/cues.npz
```

See [docs/phystwin_integration.md](docs/phystwin_integration.md) for the pinned
upstream contract, optional cue sidecar, and likelihood boundary.

Run a checkpoint-restoration parity check or a reliability-aware parameter
refit directly inside the official Warp simulator:

```bash
bpt-phystwin-refit \
  /path/to/PhysTwin final_data.pkl optimal_params.pkl best_199.pth cues.npz \
  runs/CASE/refit_mixture \
  --variant mixture \
  --train-end-frame 64 \
  --epochs 20 \
  --learning-rate 1e-3
```

See [docs/phystwin_refit.md](docs/phystwin_refit.md) for optional CUDA runtime
requirements, matched baseline definitions, provenance outputs, and current
inference limitations.

The current advanced workflow combines fit-only profile likelihoods across
interactions, recovers any available raw camera cues, fits a validation-selected
capped residual, and evaluates paired future trajectories:

```bash
bpt-combine-phystwin-profiles ...
bpt-build-phystwin-raw-cues ...
bpt-fit-phystwin-residual-dynamics ...
bpt-compare-phystwin-trajectories manifest.json comparison.json
```

See [docs/phystwin_advanced_inference.md](docs/phystwin_advanced_inference.md)
for the causal split contract, complete commands, and interpretation boundary.

## Compute

GPU experiments are intended to run on:

- `gpuserver6000`
- `gpuserver4090`

Both hosts are expected to be configured in SSH config and reachable through
the jumpserver:

```bash
ssh gpuserver6000
ssh gpuserver4090
```

See [docs/compute.md](docs/compute.md) for the current run conventions.

## Paper Repository

Notes, figures, and result artifacts are tracked separately in:

<https://github.com/FlorianPfaff/2026-07-Bayesian-PhysTwin-Paper>
