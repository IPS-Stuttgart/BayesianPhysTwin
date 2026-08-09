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

## Public-data technical-smoke gate

The first complete provider attempt, workflow run `31277475724`, retained 324 of
324 technical failures before inference. Its compact evidence showed one shared
failure signature and no successful prediction. Source inspection then isolated
a scope mismatch: the outer all-job invocation passed `resume=true` to every
fresh per-job Prob4D bundle, while the pinned Prob4D runner accepts resume mode
only after that individual bundle has written a progress journal. The source
repair now maps an empty or absent job directory to `resume=false` and preserves
`resume=true` only for a nonempty interrupted bundle. This is a technical defect
and repair, not a scientific result.

The fresh-resume retry at main revision `c4e68bf54aa4f039a1bed04cd4f2cdcc3eedfe4c`
was separately authorized and launched before this smoke gate existed. For any
later 324-job attempt, a separately versioned technical smoke must first run the
exact production stack on one public calibration-camera prefix. The job is
selected without payload or outcome access as the lexicographically smallest
SHA-256 job ID in the frozen 324-job admission. The smoke:

- re-hashes only that job's admitted RGB video and timestamp file;
- uses the frozen Prob4D, MotionCrafter, model-set, seed, and 58-frame prefix;
- verifies every generated prediction member and publishes the ordinary
  prediction seal or a retained technical-failure receipt;
- computes no track, Chamfer, calibration, or target metric; and
- keeps reserved evaluation frames, robot state, tactile data, confirmation
  payloads, and target outcomes closed.

The gate neither changes nor duplicates the already launched retry.

Its content-addressed `technical-smoke-result.json` authorizes one separately
reviewed full calibration retry only when the selected job passes and the model
is loaded exactly once. The workflow independently reloads that result and
re-hashes its exact seal or failure receipt, then checks the target-free
selection against the frozen admission, verifies the closed information
boundary, and emits `technical-smoke-gate.json`. A technical failure authorizes
nothing. The smoke is one-shot: it has no same-version resume or replacement
path. A new attempt after failure requires a source-independent
repair, a new reviewed implementation revision, and a new one-shot launcher. No
human approval or new physical capture is part of this gate; the evidence is the
already retained public Deform360 calibration prefix.

## Protected workflow

The reusable workflow is
`.github/workflows/deform360-calibration-visual-production.yml`. Pull requests run
hosted, data-free contract validation only. The initial payload execution is
requested exactly once by the reviewed
`.github/workflows/launch-deform360-calibration-visual-production-once.yml`
merge on `main`. A retry must rerun that original launcher workflow, preserving
its implementation revision and frozen admission unless a source-independent
defect is demonstrated before payload access. Such a repair requires a reviewed,
versioned launcher revision that names its failed predecessor. There is no
manual payload dispatch.

The reusable workflow also recognizes the exact future caller path
`.github/workflows/launch-deform360-calibration-visual-smoke-once.yml` only with
scope `technical-smoke` and `resume=false`. That launcher is deliberately kept
out of the implementation change: it is added in a second reviewed change only
after the smoke contracts are green, so merging implementation cannot execute
the provider.

The runner that carries Deform360 is scheduled using only the `self-hosted`
label. Before any protected-root access, the workflow additionally requires the
exact runtime name `workstation2` and a working `nvidia-smi`; a different
self-hosted machine therefore fails closed.

Run `31274946936` established the first such pre-payload failure: its parent
cleanliness check saw the two expected nested provider checkouts as untracked.
It stopped before admission download, model construction, and calibration-data
access. Retry v2 excludes exactly those two checkout paths and initializes the
compact failure-evidence root before any fallible setup step.

Run `31275886113` then verified and downloaded the frozen admission artifact but
failed during producer-environment bootstrap: Prob4D intentionally creates an
unseeded `uv` environment, while the workflow incorrectly invoked
`python -m pip`. It stopped before immutable model snapshots, retained
source-byte reads, inference, or prediction output. Retry v3 uses the same `uv`
installer selected by the frozen Prob4D bootstrap to install and check
BayesianPhysTwin inside that environment. The roster, provider revisions,
source files, and scientific configuration are unchanged.

Retry v4 binds scheduling and storage to the actual sole-runner deployment. It
changes no scientific estimator, cohort, frame range, seed, provider revision,
or threshold.

Run `31276893637` verified the frozen admission and installed the complete
producer environment, then stopped because `uv pip check` rejected the upstream
`decord==0.6.0` wheel's stale `cp36-cp36m-manylinux2010_x86_64` metadata tag on
Python 3.12. The same frozen wheel imports and loads its video runtime under
Python 3.12, and no other package inconsistency was reported. The run stopped
before model snapshots, retained source-byte reads, inference, or prediction
output. Retry v5 admits only this exact metadata exception, checks that no other
`uv pip check` output remains, and verifies all required provider imports, the
frozen decord version and wheel tag, its environment-local `VideoReader`, CPU
context, and CUDA availability before proceeding.

## Sole-runner storage contract

The reviewed workflow fixes the storage namespace to

```text
/mnt/lexar4tb/datasets/deform360
```

and requires the exact, canonical official/raw dataset directory:

```text
/mnt/lexar4tb/datasets/deform360/data-7fea8e2
```

It also registers the following adaptive-confirmation path solely as an
excluded lexical boundary:

```text
/mnt/lexar4tb/datasets/deform360/adaptive-confirmation-download-5a9c56d593462486bdd0953dcaf6f9c643bf8370
```

The calibration-only stage may inspect metadata for the official/raw directory,
but it does not require or stat the adaptive-confirmation directory. It never
descends into that path, hashes its members, enumerates its targets, or passes it
to the producer command.

The retained calibration-processed root is resolved in this order:

1. repository variable
   `DEFORM360_OFFICIAL_HUB_CALIBRATION_PROCESSED_ROOT`, when set;
2. reviewed conventional locations on the Deform360 volume and the historical
   calibration-processed location; or
3. one unambiguous `aligned` parent discovered below the storage root while
   pruning both raw roots.

Ambiguous discovery fails closed and requires the repository variable. The
processed root must be separate from both raw roots.

Large outputs and model caches remain on the Deform360 volume. Their defaults
are

```text
/mnt/lexar4tb/datasets/deform360/results/bayesian-phystwin/calibration-visual-production
/mnt/lexar4tb/datasets/deform360/caches/huggingface/hub
```

The repository variables `DEFORM360_CALIBRATION_VISUAL_OUTPUT_ROOT` and
`MOTIONCRAFTER_HF_CACHE_DIR` may select different absolute locations only within
the same Deform360 storage root. Raw, processed, output, and cache roots must be
disjoint.

Before model setup, the workflow emits a compact
`runner-storage-preflight.json` record containing the exact resolved roots,
storage capacity, official/raw device identity, runtime runner identity, and
explicit closed-boundary flags. It records that the adaptive-confirmation path
was registered without a stat operation and does not calculate recursive raw
directory sizes.

At runtime the workflow:

1. verifies the sole-runner storage contract and GPU availability without
   opening raw payloads;
2. downloads the frozen successful retained-source artifact from run
   `31272512658` by exact artifact ID, name, and digest;
3. verifies its internal `SHA256SUMS`, inventory ID, plan ID, admission ID,
   ten-object roster, and all 324 admitted camera jobs;
4. verifies clean, exact BayesianPhysTwin, Prob4D, and MotionCrafter revisions;
5. bootstraps the exact model snapshots frozen by the provider lock;
6. attempts one pinned MotionCrafter model-set construction, then executes every
   unfinished camera through a separate crash-safe Prob4D runner and progress
   journal without any reload; and
7. uploads only compact admission metadata, storage evidence, per-job seals or
   retained failure receipts, complete accounting, and environment evidence.

Before publication, the compact-copy validator re-hashes every copied receipt,
validates it as either a prediction seal or retained technical failure, and
requires exact agreement with the complete accounting record on job, object,
camera, admission, implementation, provider, MotionCrafter, model-set, status,
and output location. Cross-wired or duplicated receipts therefore cannot pass
merely because each individual JSON document is internally valid. The compact
artifact's canonical `SHA256SUMS` includes the resulting validation record.

The same check can be repeated after artifact download without opening any
prediction array:

```bash
python scripts/science/execute_deform360_calibration_visual_production.py \
  validate-bundle compact/production/visual-production-result.json \
  --run-root compact/production \
  --admission compact/admission/calibration-visual-execution-admission.json
```

Under `technical-smoke`, step 6 executes only the frozen selected job, the
compact accounting is `technical-smoke-result.json`, and a nonpassing result
fails the workflow's advancement gate after compact evidence is uploaded.

Large prediction arrays remain under the protected persistent output root. They
are not copied into GitHub artifacts.

After successful production, the exploratory outputs remain source evidence,
not claim-bearing observations. The separately documented
[`deform360_prob4d_source_calibration.md`](deform360_prob4d_source_calibration.md)
stage binds causal metric-prefix residuals to these prediction manifests and
fits physical-object-balanced point and gauge covariance artifacts. It cannot
authorize confirmation access.

## Causal execution contract

For every job, the producer first re-hashes both retained inputs:

- `undistorted.mp4`;
- `aligned_timestamps.txt`.

All admitted retained source files are verified before the first model
invocation. Any missing, changed, non-regular, or symlinked source aborts the
complete run before partial scientific output is produced.

Every generated Prob4D run configuration binds:

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

The persistent run directory is keyed by the execution-admission ID and exact
BayesianPhysTwin implementation revision. Re-running the same revision with
`resume=true`:

- reuses a completed seal only when every execution identity still matches;
- preserves a prior technical-failure receipt instead of silently retrying or
  replacing it; and
- permits Prob4D to resume an interrupted bundle only when its crash journal
  matches the exact run-spec hash.

The top-level resume request therefore does not force Prob4D's narrower resume
mode onto a fresh job. A missing or empty per-job output starts normally;
Prob4D receives `resume=true` only after that job has produced bundle state for
the crash-safe runner to validate.

Technical-smoke output uses a separate `technical-smoke-v1` namespace. The
smoke CLI rejects `resume=true` before any payload access and refuses any
existing same-revision smoke directory before re-reading retained payloads.

A process-level file lock prevents concurrent writers while allowing automatic
release after cancellation or runner failure. The shared adapter changes only
the per-job video, output directory, seed, and causal frame bounds. A change to
any fixed model or inference setting fails closed before adapter reuse.

## Command

The protected workflow invokes the operational CLI in this form:

```bash
python scripts/science/execute_deform360_calibration_visual_production.py run \
  --admission visual-execution-admission.json \
  --visual-provider-lock visual-provider-lock.json \
  --model-set-binding motioncrafter-model-set.json \
  --retained-root /resolved/calibration-processed/aligned \
  --output-root /mnt/lexar4tb/datasets/deform360/results/bayesian-phystwin/calibration-visual-production \
  --prob4d-root /exact/Prob4D \
  --motioncrafter-root /exact/MotionCrafter \
  --cache-dir /mnt/lexar4tb/datasets/deform360/caches/huggingface/hub \
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

The separately launched smoke uses the same arguments with the `smoke`
subcommand and without `--resume`. For that command, exit code `0` means the
single provider job and integrity checks passed, `3` means a retained technical
failure with no retry authorization, and `2` means a structural or custody
failure.

## Downstream observability handoff

Visual production does not invent the physical-state Jacobian, contact mapping,
contact covariance, or anchor-bias prior. Those quantities are produced by the
[calibration factor materializer](deform360_calibration_factor_materializer.md),
which consumes only public calibration-prefix measurements, requires calibrated
claim-bearing Prob4D input, reduces raw taxels to grouped kinematic contact
patches, and binds the result to the prediction seals. Once each physical object
has either:

- a visual-reference marginal precision matrix, visual-plus-contact marginal
  precision matrix, contact-anchor artifact, and the shared physical-query
  Jacobian; or
- a retained technical-failure row,

the existing atomic ten-object observability batch can be executed. Adaptive-
confirmation and confirmation payload access remain forbidden until the complete
Stage-1 evidence ledger and the corresponding opening authorization have passed.
