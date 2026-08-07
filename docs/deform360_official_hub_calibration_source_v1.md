# Official-Hub Deform360 calibration-source execution v1

## Purpose

This execution is the first payload-opening step of the locked
`deform360-official-hub-visuotactile-v1` study. It opens and prepares only the ten
Stage-0 calibration objects. It does not open any of the twelve confirmation
objects and does not yet fit or seal the eight calibration artifacts required for
confirmation access.

The immediate question is deliberately technical and empirical:

> Do at least eight of the ten locked calibration objects, including at least
> four sheet and four volumetric objects, provide exact selected-episode RGB,
> tactile sidecars, synchronized camera streams, recoverable robot state, and an
> action-only 81-frame window under the pinned official processing code?

A negative answer is retained without object replacement. A positive answer
permits the subsequent calibration-only MotionCrafter, Prob4D, contact-anchor,
physical-response, guard, and interval selection.

## Immutable inputs

The execution binds:

- Stage-0 selection:
  `protocols/locks/deform360_official_hub_visuotactile_v1_selection.json`;
- official dataset revision:
  `f804696d7a133908c7497ffdab43819d879b5cbc`;
- official processing repository:
  `lhy0807/deform360`;
- official processing revision:
  `d8522a4403b766aeb387510c04e89032a56fdf35`;
- visual-provider lock:
  `b04341bf8c5e9f5250b87e35f1428bd21d5b79507e4e0c27ec24226e244befaf`;
- source protocol:
  `protocols/deform360_official_hub_calibration_source_v1.json`.

The exact calibration units are the five sheet and five volumetric objects in
Stage 0. The twelve confirmation object identifiers are used only as a denylist;
their repository subtrees must not appear under the dedicated calibration root.

## Three information stages

### 1. Names-only plan

The planner queries the exact official-Hub revision and lists only the ten locked
calibration object subtrees. For each object it selects:

- `metadata.json`;
- every file in `calibration_refined/`;
- the exact MP4/timestamp pair at the registered episode index for every camera
  stream;
- the exact raw NPY/timestamp pair at that episode index for every tactile
  stream; and
- one released `median_*.npy` baseline for each admitted tactile sensor.

When a sensor exposes one baseline, that baseline is retained exactly. When it
exposes several timestamped baselines, the planner selects the unique baseline
whose filename timestamp is nearest to the selected tactile recording timestamp.
The association fails closed unless the nearest baseline is at most ten minutes
away, is separated from the runner-up by at least one minute, and the selected
baselines across sensors form one capture cluster spanning at most five seconds.
These limits and the association rule were fixed after the first names-only plan
revealed multiple baselines but before any calibration payload byte was opened.
The plan records every timestamp, distance, margin, and cross-sensor span.

Episode identity follows the official implementation: exact-stem data/timestamp
pairs are sorted by data filename and addressed by zero-based episode index.
Audio payloads are excluded. No file bytes are opened in this stage.

The names-only plan passes only with at least eight planned objects and at least
four in each stratum. Unsupported objects remain in the plan and are not
replaced.

### 2. Exact calibration download

Only the content-addressed paths in the plan are downloaded, always at the exact
dataset revision. Each local file is required to:

- resolve to its planned path beneath the dedicated calibration root;
- be a regular non-symlink file;
- match the repository-declared size where available; and
- match the Hugging Face LFS SHA-256 when the repository exposes one.

Every local file is hashed. The selected `metadata.json` hash must equal the hash
already committed in Stage 0. The resulting manifest records all downloaded
paths, sizes, repository identities, and local SHA-256 values.

### 3. RGB, tactile, robot, and action-window preparation

The exact selected recording is materialized as the sole recording in a
synthetic raw object tree. It therefore has synthetic episode index zero for the
official processing functions, while every manifest retains its original object
and episode identity. Files are hard-linked when possible and copied otherwise;
source bytes are unchanged.

The pinned official implementation then performs:

1. camera undistortion and synchronization using every selected camera stream
   covered by the released refined calibration;
2. exact tactile normalization and nearest-timestamp alignment from the released
   raw NPY, timestamp, and median sidecars;
3. ArUco robot-state recovery with the registered episode's original bimanual
   metadata; and
4. action-only selection of one 81-frame window using only
   `robot.actions` and `robot.openings`.

The action-window rule is the already merged source-only rule: maximize mean
closed-weighted gripper path length over the registered candidate grid, breaking
ties at the earliest start. It does not inspect object geometry, tactile values,
future reconstruction, tracking, or target metrics.

## Support and failure accounting

A source is prepared only when it has:

- at least eight aligned calibrated camera streams;
- at least one exactly aligned tactile stream;
- a valid recovered robot state;
- at least 81 aligned frames; and
- one valid action-only staging window.

The aggregate support gate is:

- at least 8 of 10 independent physical objects; and
- at least 4 of 5 objects in each registered stratum.

Every technical failure is retained with its last completed stage and exception.
There is no replacement, target-informed exclusion, or fallback object.

## Workflow trust and runner-capacity boundary

The source-contract gate no longer depends on GitHub-hosted runner capacity. It
runs on `workstation2` for trusted pushes, manual dispatches, and pull requests
whose head branch belongs to this repository. Pull requests from forks are not
admitted to the self-hosted runner. The contract job checks out the exact
reviewed revision with read-only credentials and installs into a fresh isolated
`RUNNER_TEMP` target site without a package cache. It opens no dataset root or
payload.

The empirical job uses a separate fresh target site rather than the runner's
Python toolcache or `venv` support. Both target sites are removed after the job,
while raw and processed calibration data remain only in their registered
persistent roots.

The empirical preparation job still has the explicit
`github.event_name != 'pull_request'` guard. Therefore no pull request can run
the names-only planner, download calibration bytes, open camera or robot data,
or inspect a confirmation-object subtree. A merge that changes this registered
lane runs the self-hosted contract gate first and only then starts the locked
calibration-source preparation.

## Persistent and uploaded data

Raw and processed calibration payloads remain on `workstation2` in dedicated
persistent roots. They are not committed or uploaded. The workflow uploads only
compact evidence:

- the names-only plan;
- the exact download manifest;
- the source-preparation result;
- console logs;
- package and runner identities; and
- GPU identity.

The repository checkout must remain clean. The dedicated calibration root is
checked for all twelve confirmation object identifiers after every run.

## What a successful run authorizes

A successful source-preparation run does **not** authorize confirmation access.
It authorizes only the next calibration-only stage on the sealed 81-frame
windows:

1. run the frozen MotionCrafter producer;
2. fit Prob4D visual reliability, conditional point covariance, and complete
   joint gauge handling;
3. fit the contact feature, grouping, linearization, covariance, and bias prior;
4. select normalized-evidence semantics;
5. calibrate physical response and nonlinear closure limits;
6. calibrate the baseline-relative regret guard;
7. calibrate the group-balanced conformal interval; and
8. serialize all eight registered artifacts plus the complete evidence-use
   ledger with `bpt experiment run seal-deform360-calibration`.

Only the resulting content-addressed confirmation-opening token may authorize the
single confirmation execution.

## Claim boundary

This execution supports only calibration-source acquisition and synchronization
claims. It is not evidence of provider competence, physical-query improvement,
tactile benefit, calibrated uncertainty, confirmation transfer, or state of the
art.
