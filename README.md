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
- capped persistent and robust Bayesian endpoint discrepancy anchors
- sparse spring-graph discrepancy smoothing and covariance solves
- raw camera/mask cue recovery and paired moving-block evaluation
- automatic dense MotionCrafter-to-spring-graph association and view gating
- selective release-archive retrieval and physical-object clustered bootstrap
- reproducible experiment configs and remote-run scripts

## Repository Layout

```text
src/bayesian_phystwin/   reusable Python package
src/causal4d/            independent counterfactual world-model benchmark
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
python3 -m pip install -e ".[dev,data,graph]"
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

Run the independent Causal4D milestone without changing the Bayesian PhysTwin
pipeline:

```bash
causal4d-counterfactual-benchmark \
  --seeds 0:5 \
  --output-dir runs/causal4d-counterfactual-v1
```

This evaluates generative-only, physics-only, and hybrid predictors on one
untouched action per rope, cloth, and soft-block object under matched and
shifted contact worlds. See
[docs/causal4d_counterfactual_benchmark.md](docs/causal4d_counterfactual_benchmark.md)
for the locked split, information boundary, metrics, and artifact schema.

Run the next Causal4D milestone, which infers realized contact on a topology
excluded from contact-model fitting:

```bash
causal4d-latent-contact-benchmark \
  --seeds 0:5 \
  --require-gates \
  --output-dir runs/causal4d-latent-contact-v1
```

The model marginalizes graph contact location, transmission gain, delay, slip,
control-frame bias, and physical parameters before intervention, then updates
their joint posterior from the first 20% of motion. See
[docs/causal4d_latent_contact_inference.md](docs/causal4d_latent_contact_inference.md)
for the transfer protocol, oracle controls, and pre-registered success gates.

The real-backend milestone replaces the controlled simulator with official
PhysTwin Warp rollouts, crosses Causal4D contact/action hypotheses with saved
Bayesian-PhysTwin parameter particles, and uses MolmoMotion trajectories only
as robust ranking evidence over those physical futures. It includes known,
hidden, and ambiguous future-action settings plus shuffled and generic language
controls. See
[docs/causal4d_phystwin_molmo.md](docs/causal4d_phystwin_molmo.md) for the
three-environment artifact pipeline and information boundary.

The complete Causal4D architecture now exports particle-specific endpoint
state and discrepancy beliefs, abduces persistent actuation and factual
contact variables, applies explicit `do(u_cf)` queries with fresh-contact
semantics, separates physical and language-conditioned posteriors, gates
MolmoMotion trust on source validation and target OOD diagnostics, and supports
constrained receding-horizon replanning. See
[docs/causal4d_abduction_intervention_prediction.md](docs/causal4d_abduction_intervention_prediction.md)
for the artifact contracts, commands, audited results, and claim boundary.

The first Causal4D paper is deliberately narrower than the complete software
architecture: **Bayesian abduction of realized interventions for counterfactual
prediction of deformable-object dynamics.** Its locked core/optional/application
hierarchy and current evidence status are in
[docs/causal4d_paper_scope.md](docs/causal4d_paper_scope.md). MolmoMotion is
excluded from the main claim, and closed-loop planning remains an application
without genuine robot execution.

The next real-data work package is a preregistered 36-execution protocol on the
same sloth instance. It crosses three registered contacts with four command
profiles and three replicate blocks while preserving 18 two-command
same-grasp sessions. Generate acquisition templates and validate the locked
split with:

```bash
causal4d-real-protocol validate-protocol \
  configs/causal4d/sloth_multi_action_v1.json

causal4d-real-protocol scaffold \
  configs/causal4d/sloth_multi_action_v1.json \
  /path/to/causal4d-sloth-multi-action-v1
```

See
[docs/causal4d_same_object_multi_action_protocol.md](docs/causal4d_same_object_multi_action_protocol.md)
for the contact/action design, slip gate, required actuator measurements,
matched-reset semantics, and leave-one-contact-and-action-out calibration
folds. Physical acquisition is not yet claimed complete.

The post-oracle real-undercoverage audit is documented in
[docs/causal4d_real_undercoverage.md](docs/causal4d_real_undercoverage.md).
Full 81-particle support raises coverage only from 50.6% to 55.1%. A
low-frequency graph-persistent discrepancy improves track error to 23.1 mm and
coverage to 67.8%, but a locked affine scale from one held-out source action
transfers harmfully. The real posterior therefore remains explicitly
uncalibrated pending an expanded multi-action calibration protocol.

The next simulator-discrepancy milestone now injects a bounded,
graph-regularized frame/rest-geometry correction inside PhysTwin and reruns
Warp. It uses only pre-holdout evidence and retains output, frame, state, rest,
and attachment-geometry ablations. Development evidence is mixed: the physical
update helps `single_lift_sloth` substantially but fails on
`double_stretch_sloth`, so it is not promoted to a confirmatory method. The
locked 12-fold same-object analysis gate is ready, while acquisition remains at
`0/36` completed manifests. `causal4d-evaluate-rest-geometry` produces the
per-frame-mode Warp summaries; `causal4d-rest-geometry-candidate-evidence` strips
them to pre-holdout scores; and `causal4d-rest-geometry-cross-action` selects 12
fit-only fold locks plus the 36 factual, 18 same-grasp, and 12 new-contact
records. Persistent source corrections are exported with
`causal4d-rest-geometry-source-correction`, executed on targets with
`causal4d-phystwin-rest-geometry-transfer`, and aggregated only when all 66
records pass `causal4d-rest-geometry-protocol-result`. Audit acquisition with
`causal4d-rest-geometry-protocol`. Before collection,
`causal4d-build-phystwin-canonical-graph` freezes the object-only graph and
`causal4d-rest-geometry-register-graph` locks it into the dataset; material
nodes/topology must stay fixed while controller attachments may vary. The notes
and exact development artifacts live in the
[Bayesian-PhysTwin paper repository](https://github.com/FlorianPfaff/2026-07-Bayesian-PhysTwin-Paper/blob/main/paper_notes/causal4d_rest_geometry_v1.md).

The follow-up structural milestone now has a separate opt-in path. It fits a
hierarchical MAP decomposition of object-persistent rest geometry, session
frame/settled state, and gravity using rigid-free graph ranks `4/8/16`, then
exports the exact corrected embedded geometry and rest lengths consumed by
Warp. `bpt-structural-recovery-benchmark` runs the controlled mechanism and
leakage gates; `bpt-diagnose-phystwin-structure` builds official-Warp
sensitivities and reruns the physical ladder; and `bpt-structural-protocol`
installs the measurement-only amendment before acquisition. The tagged
`v0.3.0-causal4d-aip` path remains unchanged. See
[docs/phystwin_structural_calibration.md](docs/phystwin_structural_calibration.md).
On the three released sloth diagnostics, baseline is selected in every case;
rest geometry changes equal-case CD/track by `+0.18%/+1.38%`, while graph
persistence remains much stronger at `-10.59%/-13.76%`. The structural
candidate is therefore not promoted before the locked multi-action protocol.

That negative result now feeds a discrepancy-localization diagnostic rather
than another structural model. A typed `DynamicDiscrepancyCorrection` uses one
frozen rank-4 graph basis, the same first six O-plus frames, the same four BPT
particles, and the same untouched continuation to compare graph-persistent
readout correction, a position-and-velocity prefix state restart, constant
generalized force inside Warp, and a matched rest-geometry negative control.
The deterministic simulator has an opt-in per-node force input whose disabled
path performs no force-buffer write; every real run must pass bitwise zero-force
parity. Released fused tracks do not identify camera-specific residuals, so the
observation audit reports cross-view tests as unavailable unless per-view 3D
identities are supplied. See
[docs/phystwin_discrepancy_localization.md](docs/phystwin_discrepancy_localization.md).
On the three released sloth diagnostics, readout persistence remains strongest
at `-17.42%/-13.52%` equal-case CD/track. Prefix state changes CD/track by
`-3.43%/+1.04%`, constant force by `-1.89%/-1.27%`, and the matched rest
control by `-1.40%/-2.66%`. Force fails every promotion gate and hits its
predeclared limit in two cases. The result localizes the useful correction to
the readout level but does not identify observation bias, because the release
lacks the per-view evidence needed to separate it from richer dynamics.

The released milestone is frozen at tag
`phystwin-discrepancy-localization-v1`. A post-hoc trajectory-only audit shows
that injected state corrections have short transients but nonzero and
case-dependent floors; this rejects the stronger claim that one common
contractive time constant explains the failed state branch. Before physical
collection, the versioned pre-acquisition amendment adds 12 source-only exact
repeat/signature executions, session-clustered inference, a factorial
readout-shrinkage gate, and nine independent calibration units per outer fold
without changing any of the 36 confirmatory target executions. It also locks
PyRecEst 2.4.1 as an optional source-only actuator synchronization diagnostic.
The superseding v3 lock adds leave-one-replicate-out mechanism fitting,
held-out correction shrinkage, exact rank-9-of-9 conformal arithmetic, and
noise-scaled discovery thresholds without adding or changing an execution. See
[docs/causal4d_preacquisition_v3.md](docs/causal4d_preacquisition_v3.md). The
v4 addendum leaves those thresholds and executions unchanged, validates the
10%/8-of-12 mechanism gate with matched placebo and positive controls, locks a
prospective mode-0/reset-scale prediction, and requires rejected contact-patch
alternatives to remain in the registration artifact. See
[docs/causal4d_preacquisition_v4.md](docs/causal4d_preacquisition_v4.md).

A separate public-only Causal4D track is prepared for PokeFlex while gated
dataset access is pending. It pins the public loader revision, validates raw or
processed episode schemas, assigns object/take splits from metadata hashes, and
keeps geometry-only metrics separate from unverified material identities. No
PokeFlex outcome or raw data is in this repository. See
[docs/causal4d_pokeflex_public_readiness.md](docs/causal4d_pokeflex_public_readiness.md).

MolmoMotion now has a separate pre-beta competence gate. The corrected adapter
samples the 30 fps PhysTwin videos at the checkpoint's 15 fps rate and records
the temporal contract in every forecast artifact. On `single_lift_sloth`, the
corrected forecast still fails zero-motion, constant-velocity, motion-scale,
action-ranking, and stability gates; the true lift ranks fifth of five for all
three paraphrases. See
[docs/causal4d_molmo_acceptance.md](docs/causal4d_molmo_acceptance.md). Semantic
reweighting remains disabled with `beta=0`.

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

Regenerate the probabilities discarded by the release from the raw camera
streams, while preserving all 5,000 archived frame-zero queries per camera:

```bash
bpt-build-phystwin-cotracker3-cues \
  data/phystwin-eval/CASE/final_data.pkl \
  data/different_types/CASE \
  /path/to/co-tracker/checkpoints/scaled_online.pth \
  /path/to/co-tracker \
  runs/cotracker3-cues/CASE/cues.npz \
  --train-end-frame TRAIN_END \
  --summary-json runs/cotracker3-cues/CASE/summary.json
```

The frozen extraction uses official CoTracker revision
`82e02e8029753ad4ef13cf06be7f4fc5facdda4d`, checkpoint SHA-256
`205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218`,
and retains separate visibility/confidence probabilities, training-prefix
forward/backward disagreement, and calibrated-camera multiview reprojection
error. It never decodes a held-out frame. Run the complete cohort with
`scripts/remote/run_phystwin_cotracker3_cues.sh`, then lock cue scales on the
three development cases and evaluate them on the other 19 cases:

```bash
bpt-evaluate-phystwin-perception-cues \
  data/phystwin-eval runs/cotracker3-cues runs/perception-cue-confirmation
```

See [docs/phystwin_integration.md](docs/phystwin_integration.md) for the pinned
upstream contract, optional cue sidecar, and likelihood boundary.

Generate MotionCrafter point maps/scene flow at native frame rate, associate
them with the PhysTwin spring graph without manual identities, and select one
camera using training data only:

```bash
bash scripts/remote/run_phystwin_motioncrafter.sh CASE
bpt-select-phystwin-motioncrafter-view \
  runs/motioncrafter-selection.json \
  /path/to/CASE/camera0_native/association_frozen/summary.json \
  /path/to/CASE/camera1_native/association_frozen/summary.json \
  /path/to/CASE/camera2_native/association_frozen/summary.json
```

The selector minimizes training dense error divided by training-end graph
coverage. Sparse manual tracks are optional and are read only for a post-lock
audit. See [docs/phystwin_advanced_inference.md](docs/phystwin_advanced_inference.md)
for the pinned MotionCrafter revision, leakage boundary, controls, and current
claim limit.

Re-associate anonymous MotionCrafter positions and scene flow against one fixed
PhysTwin trajectory at every frame, then apply a bounded spring-Laplacian state
observation:

```bash
bpt-assimilate-phystwin-motioncrafter \
  /path/to/CASE /path/to/RAW_CASE /path/to/camera0_native/0.npz \
  runs/CASE/motioncrafter-assimilation

bpt-evaluate-phystwin-motioncrafter-assimilation \
  runs/motioncrafter-assimilation-evaluation \
  runs/*/motioncrafter-assimilation/summary.json
```

This is an offline reconstruction control because future MotionCrafter frames
enter the state observations. On the frozen 19-case cohort it does not improve
PhysTwin: the equal-case future manual error change is `+0.62 mm`
`[-0.96, +2.17]`, only 6/19 cases improve, and direct graph support is `3.85%`.
Do not inject this update into a predictive rollout.

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

Rich-cue refits can disable the recovered proxy fields and consume the frozen
continuous transform directly with `--disable-flow-cue`,
`--disable-boundary-cue`, `--forward-backward-scale-px`, and
`--multiview-scale-px`.

See [docs/phystwin_refit.md](docs/phystwin_refit.md) for optional CUDA runtime
requirements, matched baseline definitions, provenance outputs, and current
inference limitations.

The current advanced workflow combines fit-only profile likelihoods across
interactions, recovers any available raw camera cues, compares matched residual
dynamics, and evaluates persistent/Bayesian endpoint anchors:

```bash
bpt-combine-phystwin-profiles ...
bpt-build-phystwin-raw-cues ...
bpt-fit-phystwin-residual-dynamics ...
bpt-fit-phystwin-hierarchical-residual \
  data/phystwin-eval runs/phystwin-hierarchical-residual
bpt-compare-phystwin-residual-scales \
  data/phystwin-eval runs/phystwin-hierarchical-residual \
  runs/phystwin-residual-cap-controls \
  runs/phystwin-residual-scale-comparison.json
bpt-compare-phystwin-trajectories manifest.json comparison.json
bpt-fit-phystwin-bayesian-anchor ...
bpt-audit-phystwin-calibration \
  data/phystwin-eval runs/phystwin-calibration \
  --anchor-run-dir runs/phystwin-bayesian-anchor
bpt-analyze-phystwin-horizon \
  data/phystwin-eval runs/phystwin-confirmatory \
  runs/phystwin-baselines runs/phystwin-horizon.json
bpt-analyze-phystwin-spatial-modes \
  data/phystwin-eval runs/phystwin-spatial-modes \
  --cohort confirmation
bpt-analyze-phystwin-controller-sensitivity \
  /path/to/PhysTwin data/phystwin-eval runs/phystwin-controller-sensitivity \
  --cohort development
bpt-compare-phystwin-graph-anchors \
  data/phystwin-eval runs/phystwin-graph-development \
  --cohort development --select-prior-strength
bpt-compare-phystwin-graph-anchors \
  data/phystwin-eval runs/phystwin-graph-confirmation \
  --cohort confirmation --prior-strength 0.1 --covariance-probes 16
```

The full release evaluation subset can be fetched without downloading the
archives wholesale, and the separately released label-free cloth cohort has a
frozen confirmation command:

```bash
bpt-fetch-phystwin-eval-data data/phystwin-eval
bpt-fetch-phystwin-eval-data data/phystwin-additional --additional
bpt-confirm-phystwin-additional-anchor \
  data/phystwin-additional runs/additional-anchor
bpt-confirm-phystwin-additional-bayesian \
  data/phystwin-additional runs/additional-bayesian
bpt-confirm-phystwin-additional-anchor \
  data/phystwin-additional runs/additional-se3 --spatial-mode se3
```

The per-point and Bayesian additional-cohort methods use only released training
pseudo-measurements, apply no per-case selection, consume no future actions or
observations, and write locked protocol IDs plus paired interaction/object
bootstrap summaries.
Post-hoc endpoint controls are available through `--spatial-mode` with
`global_translation`, `se3`, `sim3`, or `affine`; each fits one training-endpoint
transform and applies it unchanged to the future trajectory under the same
10 mm cap.

The graph command compares raw tracked-point anchors, kNN lifting, and a sparse
Bayesian Laplacian posterior on the exact released object spring graph. The
development-selected `lambda = 0.1` strongly reduces spatial roughness, but its
frozen graph-over-kNN CD and track intervals cross zero; treat it as a coherent
uncertainty-bearing regularizer, not a confirmed accuracy improvement.

The calibration audit fits a fixed robust anchor only before each validation
interval, freezes it, calibrates per-case finite-sample future CD/track upper
bounds, and tests lifted anchor covariance against manual tracks with 3D NEES.
On the 19-case audit, posterior-scaled 90% track bounds reach 90.63% equal-case
future coverage, while CD reaches only 75.36%. The operational selected
posterior has mean 3D NEES 1355.05 and 38.31% nominal-90% ellipsoid coverage;
its raw variance must not be described as calibrated. The conformal guarantee
is conditional on exchangeable frame scores, an assumption tested by the saved
early/middle/late coverage readout.

The spatial-mode analysis applies the same capped endpoint anchor as a
per-point field, translation, SE(3), Sim(3), and affine transform on one paired
main-release cohort. It scores both official future metrics and records
endpoint variance explained, graph coherence, and residual concentration near
the optimized controller neighborhood and PhysTwin's `z = 0` ground plane.

The controller-sensitivity command restarts every candidate from the same
released endpoint state and applies endpoint-zero, temporally correlated,
antithetic translations to one inferred trajectory per hand. Its default
1/2/5 mm sweep plus 10 mm stress test reports object-motion gain, both official
future metrics, and a future-label oracle only as an explicitly post-hoc upper
bound. Random jitter is a sensitivity diagnostic, not a correction method.

Simulator restarts use a deterministic per-vertex spring-force kernel by
default. It sums incident springs in fixed index order and avoids the released
GPU atomic accumulation whose roundoff ordering can amplify into millimeter
trajectory differences. `--atomic-spring-forces` retains the released kernel
as a diagnostic control; deterministic runs still report repeated identical
endpoint restarts and keep frame-zero replay parity separate.

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
