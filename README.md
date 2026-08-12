# Bayesian PhysTwin

Reliability-aware Bayesian belief estimation for PhysTwin-style deformable
digital twins.

Bayesian-PhysTwin treats learned tracks, depth points, masks, flow, and related
4-D perception outputs as noisy pseudo-measurements. It combines them with a
PhysTwin physical prior while keeping observation reliability, structured
covariance, physical-parameter uncertainty, and simulator discrepancy explicit.
When an update is not identifiable or fails a prospective guard, the library
retains the physical baseline instead of silently applying an unsafe correction.

## Scientific scope

This repository owns:

- versioned observation, physical-linearization, belief, provider, and run
  provenance contracts;
- reliability-aware and structured robust likelihoods;
- recursive, gauge-aware, and prior-aware Bayesian updates;
- guarded predictive-discrepancy and fallback logic; and
- adapters and evaluation utilities for official PhysTwin artifacts.

A predictive readout-discrepancy belief is not automatically a corrected latent
physical state. Released trajectories also do not identify a unique physical
cause. Experiments and papers should preserve that distinction explicitly.

## Current evidence

On the official ordered 22-case PhysTwin cohort, the frozen Bayesian anchor
improves equal-case Chamfer distance by **12.09%** and track error by **12.78%**
relative to re-evaluated released `inference.pkl` trajectories. The result is
better than released PhysTwin under that protocol, but it is not overall state
of the art against later published methods.

The simple last-residual method is the principal matched deterministic
comparator. It is nearly tied in Chamfer distance (`0.010185 m` versus
`0.010180 m`) and marginally better in track error (`0.019156 m` versus
`0.019205 m`). Raw posterior covariance is also severely undercalibrated:
operational mean 3-D NEES is `1355.05`, nominal-90% ellipsoid coverage is
`38.31%`, and the archived conformal bounds carry median case-mean upper-bound
widths of approximately `38.87/42.68 mm` for CD/track.

A registered retrospective covariance-only analysis then preserved the exact
`last_residual` point-prediction object in all `22/22` cases and changed Gaussian
NLL by `-9.136`, with simultaneous 95% CI `[-13.961, -4.312]` and `17/22`
object-session wins. Marginal 90% coverage increased from `0.706` to `0.910`,
but mean full interval width increased from `0.01645 m` to `0.05094 m`
(`3.10×`). This is retrospective uncertainty-mechanism evidence with a material
width cost; it is not calibrated raw covariance, improved point prediction, or
independent-object transfer.

The [release-facing claim contract](docs/phystwin_release_claim_v1.md) keeps the
point result, matched comparator, raw calibration failure, retrospective
covariance-only result, conformal risk–coverage–width result, and
independent-validation boundary together. The
[full-22 evidence report](docs/phystwin_sota_22_v1.md) remains the frozen source
for the cohort, intervals, render metrics, provenance, and permitted
within-contract claim.

Independent real-provider transfer is not yet established. A retrospective
19-interaction MotionCrafter test was negative. The earlier complete-stream
official-Hub Deform360 provider version also remains terminal at its frozen
source-support prerequisite: `313/324` streams were supported, `11` support
negatives were retained, `0` technical failures occurred, source covariance was
not fitted, and its twelve confirmation objects were not opened.

A separate registered Deform360 v6 route now freezes exactly ten opened source
object-sessions and twelve disjoint confirmation object-sessions, the exact
`last_residual` mean, covariance donor `independent_endpoint_v1`, horizon scales
`[8, 16, 16]`, and unchanged physical fallback. It requires 100 sealed
prefix-only source predictions and a source authorization decision before any
confirmation payload can be opened. The paper-side analysis was preregistered
before target access. The twelve confirmation object-sessions remain closed;
there is still no fresh-transfer result.

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

[Prob4D](https://github.com/IPS-Stuttgart/Prob4D) can export the portable
`ObservationBeliefV1` contract. Bayesian-PhysTwin owns the reliability-aware
belief update and PhysTwin provider boundary.
[Causal4D](https://github.com/IPS-Stuttgart/Causal4D) separately owns abduction,
intervention, and counterfactual prediction.

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

The package installs exactly one executable: `bpt`.

<!-- bpt-stable-commands:begin -->
| Command | Purpose | Documentation |
| --- | --- | --- |
| `bpt provider manifest` | Print the Causal4D provider capability manifest. | [Guide](docs/causal4d_provider_v1.md) |
| `bpt observation validate` | Validate or summarize an ObservationBeliefV1 artifact. | [Guide](docs/observation_belief_contract.md) |
| `bpt residual replay` | Replay exported residuals through the robust likelihood. | [Guide](docs/residual_replay.md) |
| `bpt benchmark synthetic` | Run the controlled synthetic benchmark. | [Guide](docs/synthetic_benchmark.md) |
| `bpt evidence summarize` | Summarize matched guarded prospective evidence. | [Guide](docs/decisive_evidence_protocol.md) |
| `bpt evidence bundle` | Build or validate a content-addressed claim bundle. | [Guide](docs/claim_bundle_v1.md) |
| `bpt run manifest` | Create or validate a content-addressed run manifest. | [Guide](docs/reproducible_runs.md) |
<!-- bpt-stable-commands:end -->

The dispatcher imports only the selected command module. Rendering help,
listing commands, and inspecting metadata therefore do not require optional
graph, vision, data, or experiment-only dependencies.

## Research commands

Additional research functionality is organized under grouped `bpt` routes. Use
the built-in registries to discover current experiments, diagnostics, and
archived analysis protocols together with their lifecycle status and optional
dependency requirements.

```bash
# Current research protocols
bpt experiment list
bpt experiment describe confirm-phystwin-bayesian-anchor
bpt experiment run confirm-phystwin-bayesian-anchor --help

# Audits and analyses
bpt diagnostic list
bpt diagnostic describe audit-phystwin-calibration

# Archived analysis protocols and negative results
bpt archive list
bpt archive describe evaluate-phystwin-state-injection

# Complete machine-readable registry
bpt commands list --json
```

See [command-line interface](docs/command_line.md) for lifecycle definitions and
the contribution procedure.

## Common stable workflows

Replay the bundled residual example:

```bash
bpt residual replay examples/residuals_demo.csv \
  --summary-json outputs/residuals_demo/summary.json \
  --scored-csv outputs/residuals_demo/scored.csv
```

Summarize matched guarded evidence with common fallback treatment:

```bash
bpt evidence summarize \
  runs/prospective/evidence.json \
  runs/prospective/summary.json \
  --reference-method last_residual
```

Run the controlled fixed-graph benchmark:

```bash
bpt benchmark synthetic \
  --seeds 1000:1020 \
  --conditions clean,iid,correlated \
  --action-modes dynamic,quasi_static \
  --output-json runs/synthetic_v3/results.json \
  --output-csv runs/synthetic_v3/aggregate.csv
```

## Python API

New integrations should use the two versioned public namespaces rather than the
historical package-root compatibility shim.

### Portable artifacts

`bayesian_phystwin.v1` owns portable observations, physical queries, run
manifests, evidence decisions, and claim bundles. Loading an observation belief
revalidates its schema, content address, covariance, identities, and exclusive
causal cutoff:

```python
from bayesian_phystwin.v1 import load_observation_belief

belief = load_observation_belief("observation_belief.npz")
print(belief.summary())
```

### Guarded inference

`bayesian_phystwin.inference.v1` owns the supported candidate-to-guard-to-route
path for new inference consumers:

```python
from bayesian_phystwin.inference.v1 import (
    finalize_guarded_update,
    infer_prob4d_candidate,
)

candidate_inference = infer_prob4d_candidate(
    observation,
    linearization,
    physical_prediction_xyz_m=physical_prediction,
    config=frozen_solver_config,
)
result = finalize_guarded_update(
    candidate_inference,
    baseline_belief,
    candidate_belief,
    guard_decision,
    metadata={"protocol_id": protocol_id},
)

if result.exact_fallback:
    assert result.selected_belief is baseline_belief
else:
    assert result.selected_belief is candidate_belief
```

Candidate inference does not choose a guard or establish covariance calibration.
Point-mean and covariance-only candidates should remain separate registered
complete beliefs. Run the deterministic accepted/fallback demonstration with:

```bash
python examples/guarded_inference_v1.py
```

The portable artifact contract is documented in
[ObservationBeliefV1](docs/observation_belief_contract.md). The
[Guarded inference API v1](docs/inference_v1.md),
[gauge-aware adapter](docs/gauge_aware_observation_update.md), and
[prior-aware guarded update](docs/prior_aware_guarded_update.md) describe the
candidate, state-update, and exact-fallback boundaries.

## Documentation map

- [PhysTwin release-facing claim contract](docs/phystwin_release_claim_v1.md):
  mandatory point-result, comparator, calibration, retrospective covariance,
  conformal-width, and independent-validation wording for software releases.
- [Command-line interface](docs/command_line.md): grouped routes, lifecycle
  registry, and contribution policy.
- [Guarded inference API v1](docs/inference_v1.md): canonical versioned
  candidate inference, caller-owned guard, and exact complete-belief fallback.
- [Experiment and evidence index](docs/experiment_index.md): frozen reports,
  negative results, experimental command families, and placement policy.
- [Decisive evidence protocol](docs/decisive_evidence_protocol.md): matched
  risk–coverage, exact fallback, tail regressions, and calibration summaries.
- [Covariance-only value certificate](docs/covariance_only_value_certificate.md):
  exact-mean finite-group admission by proper score, interval width, and
  harmful-group probability without authorizing a point update.
- [Prospective belief updates](docs/prospective_belief_updates_v1.md):
  evidence-weighted endpoint uncertainty, strict Prob4D update composition,
  gap-aware reliability, and their empirical claim boundaries.
- [Deform360 visual-provider locks](docs/deform360_visual_provider_lock_v1.md):
  target-blind producer identity before calibration data and calibration-derived
  method locks before confirmation data.
- [Finite-group calibration design](docs/finite_group_calibration_design.md):
  independent-object rank limits, information order, and fail-closed planning.
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
