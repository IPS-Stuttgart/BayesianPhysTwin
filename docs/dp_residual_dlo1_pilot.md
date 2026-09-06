# DLO1 Dirichlet-process residual pilot

This is a **retrospective source-development experiment**, not an official
DEFORM evaluation, a fresh confirmation, or a modification of any frozen paper
result. It tests recorded trajectories only. No robot, active probing, new
interaction, or new recording is required.

Implementation: `scripts/remote/run_dp_residual_dlo1_pilot.py`.
Initial execution: [Actions run 34052958639](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34052958639).
The Actions conclusion and retained `result.json`, not this protocol document,
determine whether execution completed and what was observed.

## Data and forecast boundary

Reuse the DLO1 `train` source manifest from
`results/sota/deform_dlo_source_v1/source_manifest.json`: 40 fit trajectories,
8 validation trajectories, and 8 source-test trajectories. These source-test
trajectories have been used in earlier DLO1 method development; they are not a
fresh independent confirmation panel. Each recording has 500 frames; the
forecast starts from the first two observed states and runs for 498 steps.

The base predictor is the frozen 6,400-update DEFORM physics-learning hybrid,
not a bare physical simulator. The checkpoint, source-manifest hashes, permitted
trajectory paths, and baseline metric reproduction are checked. The existing
`r1_s0p5` residual mean and coordinate variance are also independently reproduced
before mixture selection.

All predictions use only the two observed states, known future clamped-node
motion, and the frozen baseline rollout. Future free-node values are fit/score
labels, never gate inputs. Components are shared across an entire trajectory and
all internal nodes. Clamped-node predictions remain exactly the baseline.
Official evaluation directories and DLO2--DLO5 are guarded against reads.

## Method actually implemented

1. Build the existing local-coordinate, action-conditioned residual features.
2. Construct one permitted-input summary and one training residual summary per
   fit trajectory. Fit four-dimensional context PCA and three-dimensional
   residual PCA using fit data only.
3. Fit a variational Bayesian Gaussian mixture to the seven-dimensional joint
   summaries. The DP arm uses a truncated stick-breaking prior (cap six;
   concentration 0.1, 1, or 10). The finite arms use the same mixture family with
   a finite Dirichlet prior, K in {2, 3, 4, 6}, and total concentration one.
4. Fit per-node ridge experts with soft trajectory responsibilities. The
   coefficient covariance uses trajectory-cluster scores. An expert with fewer
   than two expected/effective trajectories falls back to the global fit.
5. At prediction time, marginalize out the residual-summary coordinates and
   compute weights from context alone. Preserve the mixture instead of requiring
   hard assignment.

This is a **two-stage summary-mixture residual adapter**. It is not a jointly
inferred DP regression posterior, sticky HDP-HMM, full latent-state smoother, or
joint inference of perception covariance and dynamics. Its predictive density
uses plug-in gate parameters and the existing cluster-robust expert uncertainty
construction. It does not establish automatic calibration.

## Matched comparisons and selection

Compare the frozen hybrid baseline; fixed existing ridge=1/shrinkage=0.5;
validation-selected single ridge; validation-selected finite mixture;
validation-selected DP mixture; and a nonlinear single-expert control adding
64 deterministic random Fourier features at bandwidth 1 or 3.

Each tunable family uses ridge in {0.01, 1, 100} and shrinkage in
{0, 0.125, 0.25, 0.5, 1}. Zero shrinkage preserves the baseline point forecast;
it does not represent a complete-belief fallback claim. All families use the
same inputs and base predictor. Minimum validation coordinate L1 determines
selection, with a fixed iteration-order tie rule.

Validation candidates and fitted heads are saved and content-hashed before the
source-test trajectories are loaded. Source-test outcomes do not select an arm.
A hard-assignment version of the already-selected DP is a descriptive ablation,
not another tuned candidate.

## Outcomes and statistical units

Primary: trajectory-balanced coordinate L1 in millimetres over the released
13-node convention. Also report last-quarter L1 and coordinate RMSE. All
recordings have equal lengths; eight source-test trajectories are the units for
paired bootstrap summaries, not their individual frames or coordinates.

Secondary: marginal coordinate mixture log score (density expressed per metre),
90% marginal coverage, exact marginal mixture-quantile interval width, and
moment-based coordinate NEES on the nine free nodes. These marginal diagnostics
are not joint trajectory calibration. Mixture variance retains both within- and
between-expert terms.

Do not compare these DLO1 source numbers directly with DLO2 official-test
numbers, promote a DP advantage from a baseline-only comparison, or interpret
occupied components as recovered physical mechanisms.

## Execution and retained evidence

The branch-specific workflow runs on `gpuserver4090`, with numerical dependencies
installed into a new per-run directory rather than changing the frozen runtime.
It runs implementation self-checks, verifies baseline reproduction, fits and
selects the models, then scores the separate source-test partition.

Retained files include `protocol.json`, `validation_candidates.json`,
`selection.json`, `source_opening.json`, fitted pickle-free NPZ files,
`source_predictions.npz`, `result.json`, and the execution log. Technical failures
are retained as failures, not converted into numerical evidence.

On a compatible source-data runner:

```bash
PYTHONPATH=src:scripts/remote:. python scripts/remote/run_dp_residual_dlo1_pilot.py --self-test
PYTHONPATH=src:scripts/remote:. python scripts/remote/run_dp_residual_dlo1_pilot.py \
  --output-root /new/empty/dp-pilot-evidence
```

Final numerical interpretation belongs in the private paper repository. This
pilot does not authorize changes to existing claims, protocols, or results.
