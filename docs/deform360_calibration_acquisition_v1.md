# Deform360 calibration acquisition v1

## Purpose

The official-Hub Deform360 study separates three information boundaries:

1. Stage 0 selects exact object/episode units from object names and
   `metadata.json` only.
2. The visual-provider freeze fixes the executable Prob4D/MotionCrafter producer
   before any selected raw payload is opened.
3. Stage 1 may open the ten locked calibration units, but every confirmation
   camera, tactile, robot, geometry, and target payload remains closed until all
   eight registered calibration decisions are sealed.

This acquisition lane implements the first empirical part of Stage 1. It creates
reviewable source custody and a calibration-only evidence ledger. It does not fit
or select the eight calibration artifacts and does not issue a confirmation
opening token.

## Locked cohort

The command derives the cohort from
`protocols/locks/deform360_official_hub_visuotactile_v1_selection.json`; callers
cannot supply object IDs or episodes.

| Stratum | Object | Episode |
| --- | --- | ---: |
| Sheet | `167-glove-gray-cloth` | 0 |
| Sheet | `198-kneepad-cloth` | 2 |
| Sheet | `026-sock-cloth` | 7 |
| Sheet | `031-cotton-cloth` | 0 |
| Sheet | `036-napkin-cloth` | 9 |
| Volumetric | `153-cake` | 5 |
| Volumetric | `152-slime` | 8 |
| Volumetric | `186-monster` | 6 |
| Volumetric | `058-roll-napkin` | 1 |
| Volumetric | `193-frog` | 7 |

The twelve confirmation object IDs are copied into the acquisition plan only as
a forbidden set. They are never accepted as command-line inputs and never listed
or downloaded by the runtime.

## Exact payload allowlist

Before payload access, the runtime lists only each selected calibration object's
repository subtree at the frozen dataset revision. It then writes a
content-addressed `payload-allowlist.json`. For every object the allowlist
contains only:

- `metadata.json`;
- refined intrinsics, extrinsics, and distortion files;
- one exact MP4/timestamp pair per official camera, selected by the pinned
  Deform360 exact-stem episode ordering;
- one exact NPY/timestamp pair per tactile sensor when available; and
- every `median_*.npy` tactile baseline required by the official processor.

WAV and FLAC audio, other episodes, geometry, reconstruction, depth, tracking,
point clouds, control points, and target artifacts are excluded. A dedicated
data root fails closed if it already contains an unselected file, a confirmation
object, an unexpected object, a symlink, or a missing selected file. The freshly
downloaded `metadata.json` hash must equal the Stage-0 hash.

Transport failures abort the run and may be retried against the same immutable
allowlist. They do not authorize object replacement. Once a complete source set
exists, each selected object receives exactly one processing disposition.

## Pinned official preparation

The workflow checks out `lhy0807/deform360` at
`d8522a4403b766aeb387510c04e89032a56fdf35` and verifies that the imported Python
package originates from that checkout. For each unit it runs, in order:

1. `deform360.undistort.undistort_episode` for all official cameras;
2. `deform360.tactile.process_tactile_episode` for all exact tactile streams; and
3. `deform360.processing.robot_stage.process_robot_episode`, using the released
   per-episode `bimanual` metadata and root seed zero.

The selective local mirror contains exactly one source recording pair per
sensor. Because the official library assigns episode indices from recordings
present in the local directory, the runtime invokes those pinned stages with
local processing index zero. Every case separately records the original locked
source episode and `official_processing_episode_index=0`; source file names and
hashes retain the exact original episode identity. No earlier episode payload is
downloaded merely to preserve an array index.

A successful case records only hashes, frame count, camera count, tactile sensor
count, bimanual status, and the aligned timeline identity. Raw and processed
payload bytes remain in the dedicated self-hosted data root.

## Failure and denominator policy

A selected object is never silently dropped or replaced. An official processing
failure produces a `technical_failure` case with:

- the complete selected raw-file hash inventory;
- the failing stage and exception type;
- a SHA-256 of the normalized failure message;
- hashes of partial outputs, when any; and
- an explicit `technical_failure_retained_without_replacement=true` marker.

The evidence ledger still contains one `calibration_only` entry for that object.
Consequently, all ten locked objects remain represented in later calibration
selection and reporting. This is distinct from retrying a transport failure for
the same immutable source bytes.

## Output

The compact workflow artifact contains no MP4, NPY, NPZ, PNG, PLY, or PCD data.
It contains:

```text
acquisition-plan.json
payload-allowlist.json
download-manifest.json        # content-addressed downloaded-byte inventory
cases/*.json
evidence-use-ledger.json
acquisition-result.json
failures/*.txt          # only when official processing fails
STATUS.md
SHA256SUMS
console.log
environment.txt
```

The result's source-artifact map also binds the exact protocol, Stage-0
selection, visual-provider lock, acquisition contract module, and runtime-script
bytes. These are small provenance inputs; no raw or processed payload is copied
into the compact evidence tree.

The acquisition result records:

```text
calibration_payloads_opened=true
confirmation_payloads_opened=false
target_outcomes_used=false
replacement_allowed=false
```

The result is complete when all ten objects have either `prepared` or
`technical_failure` status. “Complete” here means complete accounting, not
successful calibration or empirical competence.

## Workflow execution

The pull-request job runs only contracts and opens no payload. After review and
merge, dispatch `.github/workflows/deform360-calibration-acquisition.yml` from
`main` with `open_calibration_payloads=true`. The data root is resolved from, in
order:

1. the dispatch `data_root` input;
2. repository variable `DEFORM360_CALIBRATION_DATA_ROOT`; or
3. `$HOME/.cache/bayesian-phystwin/deform360-official-hub-calibration-v1`.

The root must be outside the Git checkout and dedicated to this protocol.
Repository permissions are read-only. Model/data bytes are not uploaded.

## Next admissible step

Use the frozen acquisition plan, ten case records, and evidence-use ledger to
fit and select exactly the eight registered calibration roles:

1. contact feature and grouping;
2. contact linearization and covariance;
3. anchor-bias prior;
4. visual reliability and gauge;
5. normalized evidence;
6. physical response and closure;
7. regret guard; and
8. conformal interval.

Those choices must be assembled into the existing complete calibration bundle
and Stage-1 execution seal before any confirmation payload is opened.

## Claim boundary

This lane establishes source identity, exact cohort custody, official source
preparation, technical-failure accounting, and information order. It establishes
no observation quality, physical-query accuracy, tactile benefit, predictive
calibration, material identification, Causal4D improvement, deployment safety,
or state of the art.
