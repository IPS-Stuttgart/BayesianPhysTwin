# Object-Disjoint MatPhys + Bayesian-PhysTwin Evaluation

## Initialization correction

Protocol v1 was stopped before any fold completed or future was opened. Its epoch-200
warm start records `video_allcases` in the checkpoint metadata. Resetting the residual
spring heads preserved the identity arm but could not remove benchmark-object
information from the learned trunk, so v1 was not genuinely object-disjoint.

Protocol v2 is authoritative. Every fold uses the generic frozen
`MCG-NJU/videomae-base` representation and freshly seeded trainable projectors, material
codes, geometry encoders, and spring heads. The base seed is 42 and the pinned upstream
DDP rule uses `seed + rank`; both are recorded in every training audit. No checkpoint
trained or fine-tuned on a PhysTwin benchmark case is loaded. The invalidation record is
`results/sota/matphys_loo22_v1_initialization_invalidation.json`.

## Question

Can an object-disjoint, source-trained MatPhys spring proposal improve the official
PhysTwin future-prediction benchmark when it is replayed in the pinned PhysTwin/Warp
simulator and guarded by Bayesian-PhysTwin's prefix-only selector?

This is a retrospective full-benchmark study. The benchmark informed method
development, so it is not an untouched external confirmation. The leave-one-physical-
object-out training boundary and sealed future opening prevent target-object outcome
leakage within this run.

## Fixed design

- Cohort: the official 22 PhysTwin cases grouped into 11 physical objects.
- Training: one freshly initialized MatPhys model per held-out object, using every
  other object.
- Target evidence: the first 75% of the released training prefix only.
- Spring arms: exact incumbent plus log-space proposal strengths 0.25, 0.50, 0.75,
  and 1.00.
- Simulator: pinned upstream PhysTwin/Warp, with released global and contact
  parameters and no optimization after spring injection.
- Within-arm adaptation: the frozen Bayesian anchor and residual baselines compete
  using only the fit/validation prefix.
- Across-arm selection: equal mean of validation CD and track ratios to the exact
  incumbent; at least 1% improvement, no regression in either metric, and a 1 mm
  identity-replay tolerance are required.
- Tie break: exact incumbent first, then the lowest accepted proposal strength.

The validation track ratio uses the released manual 3D tracks on the permitted past
prefix. This makes the study a causally separated, online-supervised benchmark test,
not a label-free deployment result. No manual track at or after `train_end_frame` is
available to fitting or selection. A label-free selector must be frozen and evaluated
on a separate future-opening protocol rather than retrofitted after this run.

The authoritative machine-readable protocol is
`configs/sota/matphys_guarded_bayesian_loo22_v2.json`.

## Information boundary

MatPhys may use complete trajectories from source objects and target video frames only
through the registered evidence boundary. PhysTwin may roll forward under the released
future controller trajectory because this action is part of the benchmark prediction
condition. Before family selection is sealed, the replay path must not score future
object points, manual tracks, Chamfer distance, coverage, or calibration.

`bpt-phystwin-refit --selection-only` generates the complete action-conditioned rollout
but truncates every observation-based diagnostic at `train_end_frame`. The subsequent
Bayesian overlays and family gate also run in selection-only mode. Every complete
trajectory and selected family is hash-bound before the future evaluator is invoked.

## Multi-host provenance

Each completed training host runs:

```bash
python scripts/remote/build_matphys_loo_spring_fields.py collect \
  WORKSPACE/loo_workspace_manifest.json PARTIAL_BUNDLE --folds 0,2,4
```

The portable bundles retain the protocol, training-audit and export-manifest bytes,
copy every candidate spring field, and replace host-specific absolute field paths with
relative hash-bound identities. Disjoint bundles are merged on the evaluation host:

```bash
python scripts/remote/build_matphys_loo_spring_fields.py merge \
  FULL_BUNDLE PARTIAL_BUNDLE_A/loo_spring_fields.json \
  PARTIAL_BUNDLE_B/loo_spring_fields.json
```

The sealed replay and selection stage is:

```bash
python scripts/remote/run_matphys_loo_strength_sweep.py \
  FULL_BUNDLE/loo_spring_fields.json SWEEP_OUTPUT \
  --python BPT_GPU_PYTHON \
  --official-repo PINNED_PHYSTWIN_REPO \
  --data-root CONFIRMATORY_DATA \
  --cues-root COTRACKER3_CUES \
  --gpu-ids 0,1
```

This command stops after writing `family_selection/backbone_family_selection.json`.
Future metrics are opened separately with `bpt-open-phystwin-backbone-family-future`.
The strictly post-opening report is then generated with:

```bash
bpt-report-matphys-loo-sota DATA_ROOT \
  SWEEP_OUTPUT/family_selection/backbone_family_selection.json \
  FUTURE_OUTPUT/backbone_family_future.json \
  FUTURE_OUTPUT/matphys_loo_sota_report.json
```

The reporter verifies that the opener names the exact SHA-256 of the sealed selection,
then computes the predeclared horizon, worst-case, and physical-object-clustered paired
analyses. It cannot alter or rerun family selection.

## Decision rule

The point-estimate SOTA gate passes only if the selected equal-case full-22 means are
both below MatPhys's published rounded values:

- future Chamfer distance: 0.008 m;
- future manual-track error: 0.015 m.

Report the unguarded strength arms, exact incumbent, selected family, selection counts,
early/middle/late errors, worst-case regressions, and paired physical-object-cluster
uncertainty. Calibration and NEES are separate claims; a better point estimate is not
evidence of calibrated uncertainty.

If the gate fails, do not tune on the opened futures. Diagnose the locked result and
move the next proposal to an independent dataset or a newly registered protocol.
