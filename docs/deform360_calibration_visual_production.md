# Deform360 calibration visual production

## Purpose

This stage executes the frozen calibration-camera work list against the exact
retained bytes already admitted by the successful metadata custody run. It does
not rebuild the object split, regenerate the visual plan, select preferred
cameras, replace failures, open confirmation objects, or inspect target
outcomes.

The authoritative input is GitHub Actions artifact `9026183221` from run
`31272985733`:

- artifact name:
  `deform360-calibration-retained-source-admission-31272985733-1`;
- archive SHA-256:
  `d13a3aed7b63effab637215feee15c61d9cb69330dbe8f666a6e37b00b35b836`;
- inventory ID:
  `0cbf109f8ea572846f0d117880a6048840346d7b32fef7f47ba2c55f857f744c`;
- plan ID:
  `c30d2ecdbf2702460171ec6c58f2cc9ae2c666b4e311f2517b539cdc14f2eea7`;
- admission ID:
  `4dd68e209b4c1a206a209786f57b0a4a96bd102a79a0f8f60d436fabd5d584ba`;
- admitted physical objects: `10`; and
- admitted object/camera jobs: `324`.

The retained-source custody run had to inspect calibration RGB, timestamp,
tactile, and robot-state bytes to establish their identities. Visual production
does **not** repeat that materializer. It consumes the audited artifact and opens
only the admitted calibration video and timestamp files needed by each visual
job. Tactile arrays, robot-state arrays, confirmation objects, and target
outcomes remain closed in this stage.

## Protected workflow

The workflow is
`.github/workflows/deform360-calibration-visual-production.yml`. Pull requests run
hosted, data-free contract validation only. Payload access requires a manual
dispatch of a reviewed `main` revision on the protected `workstation2` runner
with labels `self-hosted`, `Linux`, `X64`, and `nvidia-smi`.

Before protected execution, a hosted job queries GitHub's artifact metadata and
fails unless there is exactly one live artifact with the pinned run ID, artifact
ID, name, archive digest, branch, and source revision. The self-hosted job then:

1. downloads that artifact by numeric ID with digest mismatch configured as an
   error;
2. rejects symbolic links and unexpected members;
3. verifies every entry in its internal `SHA256SUMS` file and the separately
   pinned inventory, plan, admission, and receipt hashes;
4. verifies all terminal IDs and the closed confirmation/target boundary;
5. verifies clean, exact BayesianPhysTwin, Prob4D, and MotionCrafter revisions;
6. bootstraps the exact model snapshots frozen by the provider lock;
7. executes each admitted camera job through the pinned
   `prob4d-motioncrafter` entry point; and
8. uploads only compact admission metadata, per-job seals or retained failure
   receipts, complete accounting, and environment evidence.

Large prediction arrays remain under the protected persistent output root. They
are not copied into GitHub artifacts. Run-local Python environments, downloaded
admission copies, and compact staging directories are removed after upload.

## Causal execution contract

The process-level production lock is acquired before job validation or model
execution. Immediately before each admitted job, the executor re-hashes:

- its exact `undistorted.mp4`; and
- its exact `aligned_timestamps.txt`.

A missing, changed, non-regular, symlinked, or path-escaping source aborts the
run. A resumed seal or failure receipt is accepted only after the corresponding
source bytes have been revalidated.

The complete video container is hashed to verify file identity. Prob4D's Decord
adapter, however, requests only the admitted half-open causal prefix. The command
binds:

- Prob4D revision `25d90ef7f78ba4307f4555cb636d666004e1bf66`;
- MotionCrafter revision `9cb4e9679f5f34e249945544052464ef46324bc2`;
- the exact four-model set and immutable Hugging Face revisions;
- deterministic model type;
- 320-by-640 resolution;
- 25-frame windows with overlap 8;
- the admitted per-view root seed and `derived-per-call` seed schedule; and
- only the admitted 58-frame causal prefix.

The subsequent 18 prediction frames are represented in the admission as reserved
evaluation frames. They are not decoded and are not passed to MotionCrafter.
Prob4D independently hashes the input video into its run specification, so a
change between the executor's source check and model startup is detected by the
post-run manifest validation.

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

Atomic no-clobber JSON publication and a process-level file lock prevent
concurrent or last-writer-wins replacement. A code-3 result is uploaded as a
complete negative operational record, after which the workflow deliberately
fails rather than presenting retained technical failures as a green run.

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
- `3`: complete accounting with one or more retained technical failures; and
- `2`: provenance, source-byte, schema, publication, or infrastructure failure.

Merging the implementation does not start GPU production. The first protected
execution should be treated as an operational runtime preflight before assuming
that 324 serial camera jobs fit within the workflow's six-hour limit. Model-load
reuse or a provider-level batch interface must be introduced under a new bound
execution contract if that preflight shows startup overhead is material.

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
