# Deform360 calibration visual production

## Purpose

This stage opens the retained calibration-camera bytes for the first time after
three independent metadata contracts have been fixed:

1. the prepared-source inventory records the exact retained RGB and timestamp
   files;
2. the visual-production plan fixes every object/camera job, causal frame range,
   seed, dependence group, and output location; and
3. the visual-execution admission proves that every planned job names the exact
   bytes present in the inventory.

The producer executes those admitted jobs and no others. It does not construct a
new object split, select preferred cameras, replace failures, open confirmation
objects, or inspect target outcomes.

## Protected workflow

The workflow is
`.github/workflows/deform360-calibration-visual-production.yml`. Pull requests run
hosted, data-free contract validation only. Payload access is restricted to a
manual dispatch of the reviewed `main` revision on the protected
`workstation2` runner with labels `self-hosted`, `Linux`, `X64`, and
`nvidia-smi`.

At runtime the workflow:

1. invokes the reviewed reusable retained-source admission workflow already
   merged on `main`;
2. consumes its uploaded inventory, plan, admission, content identities, and
   artifact digest without rebuilding a parallel custody path;
3. verifies clean, exact BayesianPhysTwin, Prob4D, and MotionCrafter revisions;
4. bootstraps the exact model snapshots frozen by the provider lock;
5. executes each admitted camera job through the pinned
   `prob4d-motioncrafter` entry point; and
6. uploads only compact admission metadata, per-job seals or retained failure
   receipts, complete accounting, and environment evidence.

Large prediction arrays remain under the protected persistent output root. They
are not copied into GitHub artifacts.

## Causal execution contract

For every job, the producer first re-hashes both retained inputs:

- `undistorted.mp4`;
- `aligned_timestamps.txt`.

All source files are verified before the first model invocation. Any missing,
changed, non-regular, or symlinked source aborts the complete run before partial
scientific output is produced.

The generated Prob4D command binds:

- Prob4D revision `25d90ef7f78ba4307f4555cb636d666004e1bf66`;
- MotionCrafter revision `9cb4e9679f5f34e249945544052464ef46324bc2`;
- the exact four-model set and immutable Hugging Face revisions;
- deterministic model type;
- 320-by-640 resolution;
- 25-frame windows with overlap 8;
- the admitted per-view root seed and `derived-per-call` seed schedule; and
- only the admitted 58-frame causal prefix.

The subsequent 18 prediction frames are recorded in the seal as reserved
evaluation frames and remain unopened by this stage.

## Prediction seal

A successful job is not accepted merely because the model process exits with
zero. The workflow invokes Prob4D's integrity verifier, which re-hashes every
prediction member and validates the bound run specification. BayesianPhysTwin
then independently requires exact agreement on:

- the source-video SHA-256 and byte count;
- the clean MotionCrafter commit;
- every inference setting and causal frame boundary;
- the model-set content identity and loader identity; and
- every overlap window remaining inside the causal prefix.

Only then is `prediction-seal.json` published. Its content ID binds the admitted
job, source files, provider and model identities, command identity, prediction
manifest hash, Prob4D run-spec hash, member count, and closed information
boundary.

## Technical failures and resumability

Technical failures are terminal records for the same admitted object/camera job;
they are not replacement authorizations. A failure receipt records the stage,
portable return code, hashes and byte counts of local logs, and complete lineage.
It contains no alternative object or camera.

The persistent run directory is keyed by the execution-admission ID. Re-running
with `resume=true`:

- reuses a completed seal only when every execution identity still matches;
- preserves a prior technical-failure receipt instead of silently retrying or
  replacing it; and
- permits Prob4D to resume an interrupted bundle only when its crash journal
  matches the exact run-spec hash.

A process-level file lock prevents concurrent writers while allowing automatic
release after cancellation or runner failure.

## Command

The protected workflow invokes the operational CLI in this form:

```bash
python scripts/science/execute_deform360_calibration_visual_production.py run \
  --admission visual-execution-admission.json \
  --visual-provider-lock visual-provider-lock.json \
  --model-set-binding motioncrafter-model-set.json \
  --retained-root /protected/calibration-processed/aligned \
  --output-root /protected/calibration-visual-production \
  --prob4d-motioncrafter /exact/env/bin/prob4d-motioncrafter \
  --prob4d-root /exact/Prob4D \
  --motioncrafter-root /exact/MotionCrafter \
  --cache-dir /exact/huggingface/cache \
  --implementation-revision "$(git rev-parse HEAD)" \
  --attempt-id RUN_ID-RUN_ATTEMPT \
  --resume
```

Exit codes are:

- `0`: every admitted view succeeded;
- `3`: complete accounting with one or more retained technical failures;
- `2`: provenance, source-byte, schema, publication, or infrastructure failure.

A code-3 result is a completed negative operational outcome and remains eligible
for downstream no-replacement accounting.

## Downstream observability handoff

Visual production does not invent the physical-state Jacobian, contact mapping,
contact covariance, or anchor-bias prior. Those quantities must be produced by a
separately reviewed calibration-only materializer and bound to the prediction
seals. Once each physical object has either:

- a visual-reference marginal precision matrix, visual-plus-contact marginal
  precision matrix, contact-anchor artifact, and the shared physical-query
  Jacobian; or
- a retained technical-failure row,

the existing atomic ten-object observability batch can be executed. Confirmation
payload access remains forbidden until that batch, the complete Stage-1 evidence
ledger, and the confirmation-opening authorization have all passed.
