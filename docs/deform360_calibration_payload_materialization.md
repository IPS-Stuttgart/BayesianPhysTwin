# Deform360 calibration payload materialization

## Purpose

The official-Hub Deform360 protocol has already frozen the metadata-only cohort
and the exact Prob4D/MotionCrafter visual producer. The next admissible action is
to open the ten selected calibration objects while all twelve confirmation
objects remain closed.

`materialize_deform360_calibration_payloads.py` performs the first empirical
Stage-1 action. It is deliberately narrower than the eventual visual and contact
processing pipeline: it materializes the exact selected-episode prefix inputs
needed to assess support and plan the expensive camera processing, without
opening camera video frames, robot arrays, geometry annotations, confirmation
payloads, or target outcomes.

## Why the official downloader is not used directly

The pinned Deform360 `deform360-download` command selects whole object
directories and does not expose a dataset-revision argument. The registered
study instead binds:

- dataset revision `f804696d7a133908c7497ffdab43819d879b5cbc`;
- ten exact calibration object/episode pairs;
- twelve exact confirmation object/episode pairs; and
- no replacement after payload access.

The materializer therefore calls the Hub at the exact Stage-0 revision, lists
only the ten calibration object prefixes, and reproduces the official
filename-sorted episode indexing before downloading any payload file.

## Opened calibration prefix

For each selected calibration unit, the command materializes only:

- `metadata.json`, whose SHA-256 must equal the Stage-0 lock;
- the three trusted `calibration_refined` NumPy dictionaries;
- the timestamp sidecar for the selected episode in every camera stream;
- the exact selected tactile `.npy` recording and timestamp sidecar; and
- the unique tactile `median_*.npy` baseline for each sensor.

Camera `.mp4` files are listed and content-bound in the plan but are not
downloaded. This makes the next camera-subset and processing decision reviewable
before large video access. WAV and FLAC tactile alternatives are ignored because
the official exact alignment requires raw `.npy`, timestamp, and baseline
sidecars.

Missing exact tactile inputs, incomplete calibration, or missing selected camera
recordings are retained as object-level technical failures. A Stage-0 metadata
hash mismatch, dataset-revision mismatch, dirty source checkout, or any path
under a confirmation object fails the complete run.

## Workflow information order

The workflow
`.github/workflows/deform360-calibration-payload-materialization.yml` has two
phases:

1. Pull requests run hosted source, contract, and adversarial tests only. They do
   not access calibration payloads.
2. A merge to `main`, or an explicit manual dispatch on reviewed `main`, runs the
   materialization on `workstation2`.

The self-hosted job checks out the exact official processing revision, uses a
persistent data root, and uploads only a compact manifest, logs, environment
record, and checksums. Raw Deform360 bytes remain on the runner and are never
placed in Git or a workflow artifact.

The default persistent root is:

```text
/mnt/lexar4tb/datasets/deform360_official_hub_visuotactile_v1
```

It can be replaced by the repository variable `DEFORM360_CALIBRATION_ROOT` or a
manual workflow input.

## Manifest boundary

A successful empirical materialization records:

```text
calibration_payloads_opened = true
camera_media_opened = false
camera_timestamp_sidecars_opened = true
trusted_camera_calibration_opened = true
tactile_arrays_opened = true
robot_arrays_opened = false
geometry_annotations_opened = false
confirmation_payloads_opened = false
target_outcomes_used = false
replacement_allowed = false
```

The manifest binds the Stage-0 snapshot, visual-provider lock, exact dataset and
processing revisions, reviewed BayesianPhysTwin revision, all selected and
confirmation object identities, every planned camera recording, every opened
path, and every downloaded file digest.

## Next gate

After this manifest is reviewed, the next empirical step is to freeze the camera
subset and causal prefix rule from calibration-only evidence, download and
undistort only those selected camera recordings, align tactile inputs, and then
produce the eight registered calibration artifacts and the complete
`EvidenceUseLedgerV1`. Confirmation payloads remain closed until the Stage-1
calibration execution seal emits its opening token.

## Claim boundary

This materialization is calibration-input and information-order evidence. It is
not evidence of provider accuracy, tactile benefit, physical-query improvement,
predictive calibration, material identification, Causal4D benefit, deployment
safety, or state of the art.
