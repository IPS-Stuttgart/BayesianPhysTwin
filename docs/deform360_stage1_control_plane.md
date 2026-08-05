# Deform360 Stage-1 control plane

## Purpose

The official-Hub Deform360 visuotactile protocol already fixes a fresh
metadata-only cohort, a finite-group calibration design, and the schemas for a
visual-provider lock and a complete calibration bundle. This control plane
turns those separate contracts into two explicit, content-addressed
transitions:

```text
Stage-0 selection + exact Prob4D/MotionCrafter provider
        |
        v
Stage-1 plan and calibration-access token
        |
        v
calibration objects only
        |
        v
complete calibration bundle + visual calibration lock
        |
        v
confirmation-opening token
```

The commands in this document do not download or inspect selected camera,
tactile, robot, reconstruction, geometry, or target payloads. They validate and
bind identities only. A successful command is therefore control-plane evidence,
not empirical evidence.

## Registered command

The grouped command is:

```bash
bpt experiment run prepare-deform360-stage1 --help
```

It has five operations:

- `provider-lock`: validate one claim-bearing Prob4D provider-v2 attestation and
  create the exact target-blind Prob4D/MotionCrafter lock;
- `plan`: bind the committed Stage-0 selection, provider lock, visual-provider
  amendment, and finite-group calibration design;
- `verify-plan`: require reviewed plan, provider, and selection identities before
  calibration payload access;
- `seal`: derive the visual calibration lock from a complete
  `Deform360CalibrationBundleV1`;
- `verify-seal`: independently rederive and verify the complete seal before any
  confirmation payload access.

Every persisted artifact is written atomically and refuses overwrite unless the
operator explicitly passes `--overwrite`.

## Prepare transition

First obtain a self-contained, claim-bearing Prob4D provider attestation. It
must contain an independently matched runtime revision, both covariance
calibration IDs, canonical covariance roots, and analytic `Sim(3)` composition
Jacobians. Then create the provider lock:

```bash
bpt experiment run prepare-deform360-stage1 provider-lock \
  --provider-attestation /evidence/prob4d-provider-attestation.json \
  --motioncrafter-revision <40-character-commit> \
  --model-set-id <sha256> \
  --initial-metric-frame-prior-id <sha256> \
  --root-seed 20260805 \
  --seed-policy per-object-derived-seed-v1 \
  --window-size 25 \
  --overlap 8 \
  --height 320 \
  --width 640 \
  --storage-dtype float32 \
  --max-gauge-rank 64 \
  --minimum-retained-gauge-trace 0.999 \
  --output visual-provider-lock.json
```

Create the Stage-1 plan from the committed target-blind inputs:

```bash
bpt experiment run prepare-deform360-stage1 plan \
  --provider-lock visual-provider-lock.json \
  --output stage1-plan.json
```

The plan independently recomputes and checks:

- the Stage-0 cohort selection identity;
- the Stage-0 content identity and complete artifact identity;
- the exact official dataset and processing revisions;
- five calibration and six confirmation objects per registered stratum;
- disjoint calibration and confirmation objects;
- the visual-provider amendment's selection and finite-group bindings;
- the pooled ten-object, rank-10, nominal-90% calibration design;
- the provider lock's complete content address; and
- the fact that no selected payload, confirmation payload, or target outcome has
  been opened and replacement remains forbidden.

Before selected calibration payloads are opened, commit or otherwise review the
exact plan and verify its recorded identities:

```bash
bpt experiment run prepare-deform360-stage1 verify-plan \
  --plan stage1-plan.json \
  --expected-plan-id <reviewed-plan-id> \
  --expected-provider-lock-id <reviewed-provider-lock-id> \
  --expected-selection-artifact-sha256 \
    dc1c2d192fbb841d2f0e290d77f21d697983b3f8bfbcae476e71fe902309cd82
```

Only the returned calibration-access token authorizes the registered calibration
objects. It does not authorize confirmation objects.

## Calibration seal transition

After calibration-only processing, create a complete
`Deform360CalibrationBundleV1`. The bundle must contain all eight registered
roles and retain every one of the ten calibration objects:

1. contact feature and grouping;
2. contact linearization and covariance;
3. anchor-bias prior;
4. visual reliability and gauge calibration;
5. normalized evidence;
6. physical response and nonlinear closure;
7. baseline-relative regret guard; and
8. conformal interval.

Derive the visual calibration lock and verify the complete seal:

```bash
bpt experiment run prepare-deform360-stage1 seal \
  --plan stage1-plan.json \
  --provider-lock visual-provider-lock.json \
  --calibration-bundle calibration-bundle.json \
  --output visual-calibration-lock.json \
  --summary-output stage1-seal-summary.json

bpt experiment run prepare-deform360-stage1 verify-seal \
  --plan stage1-plan.json \
  --provider-lock visual-provider-lock.json \
  --calibration-bundle calibration-bundle.json \
  --calibration-lock visual-calibration-lock.json \
  --expected-plan-id <reviewed-plan-id> \
  --expected-provider-lock-id <reviewed-provider-lock-id> \
  --expected-bundle-id <reviewed-calibration-bundle-id> \
  --expected-calibration-lock-id <reviewed-calibration-lock-id> \
  --expected-selection-artifact-sha256 <reviewed-selection-id> \
  --expected-evidence-use-ledger-id <reviewed-ledger-id>
```

The bridge groups the eight calibration references into four deterministic
identities required by `Deform360VisualCalibrationLockV1`:

- visual covariance, reliability, identity, and normalized evidence;
- contact mapping, covariance, grouping, and anchor bias;
- physical closure and deployment guard; and
- finite-group interval calibration.

It also binds the Stage-1 plan ID, calibration-access token, complete calibration
bundle ID, evidence-use ledger, and every calibration reference ID. The
`verify-seal` operation requires all six reviewed identities explicitly rather
than accepting a merely self-consistent bundle. The confirmation-opening token
is valid only for that exact bundle and the unchanged confirmation cohort.

## GitHub Actions entry point

`.github/workflows/deform360-stage1-control.yml` exposes three profiles:

- `contracts`: data-free hosted contract and command tests;
- `prepare`: self-hosted creation and verification of the provider lock and
  Stage-1 plan;
- `seal`: self-hosted derivation and verification of the calibration seal.

The self-hosted jobs use unique directories under `RUNNER_TEMP`, upload only
compact JSON and checksum evidence, and receive no dataset-root input. They do
not download Deform360 data and cannot open confirmation payloads.

Example prepare dispatch:

```bash
gh workflow run deform360-stage1-control.yml \
  --repo IPS-Stuttgart/BayesianPhysTwin \
  --ref main \
  -f profile=prepare \
  -f provider_attestation_path=/evidence/prob4d-provider-attestation.json \
  -f motioncrafter_revision=<40-character-commit> \
  -f model_set_id=<sha256> \
  -f initial_metric_frame_prior_id=<sha256>
```

Example seal dispatch:

```bash
gh workflow run deform360-stage1-control.yml \
  --repo IPS-Stuttgart/BayesianPhysTwin \
  --ref main \
  -f profile=seal \
  -f provider_lock_path=/evidence/visual-provider-lock.json \
  -f stage1_plan_path=/evidence/stage1-plan.json \
  -f calibration_bundle_path=/evidence/calibration-bundle.json
```

## Failure behavior

The control plane fails before authorization when any registered identity,
cohort member, source revision, calibration role, finite-sample rank, evidence
ledger, information-boundary flag, or content address changes. It rejects
symbolic-link source inputs, duplicate JSON keys, non-finite JSON, malformed
identifiers, incomplete calibration roles, cohort replacement, and any record
claiming prior confirmation or target access.

A failure does not authorize replacement, target inspection, or confirmation-side
repair. Correct the control-plane artifact or report the calibration stage as a
technical failure under the registered accounting.

## Claim boundary

A valid plan proves that one exact provider and one exact cohort were frozen
before calibration payload access. A valid seal proves that all registered
calibration choices were content-addressed before confirmation access. Neither
artifact proves observation accuracy, contact informativeness, physical-state
identifiability, predictive improvement, calibrated uncertainty, Causal4D
benefit, safety, or state of the art.
