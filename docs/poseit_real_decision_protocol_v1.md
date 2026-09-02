# PoseIt real-object decision validation v1

## Purpose

This protocol tests a narrow real-data claim: on previously unseen household
objects, a downstream-decision-directed policy can choose logged physical
holding-pose probes more efficiently than a task-agnostic system-identification
policy, while abstaining when no pose is certified stable.

The evaluation is a logged-policy study. It does not claim that the selector was
deployed online or that the released robot setup defines safety outside PoseIt.

## Why PoseIt fits

PoseIt contains 1,840 physical grasp cycles over 26 household objects and 16
holding poses. Each cycle records RGB-D, tactile, wrist force/torque, robot state,
and gripper force before a physical shake tests grasp stability. The paper also
reports that all combinations of grasp point, two gripper forces, and holding
pose were collected. This creates repeated action menus nested inside real
objects rather than treating individual frames as independent samples.

The protocol treats pose 1 as the mandatory reference observation. A probe
reveals only pre-shake sensory data at another logged pose. The shake label is
never revealed during selection and is used only to score the final chosen pose.

Primary sources:

- Repository: <https://github.com/Robo-Touch/PoseIt>
- Paper: <https://arxiv.org/abs/2209.05022>
- Public archive folder: <https://drive.google.com/drive/u/2/folders/1CQiMPBEVvRMrDBSIRVeuwyuUOCOesfMc>

## Frozen design

- Statistical unit: one physical object.
- Nested repeat: one grasp-location by gripper-force family.
- Fit/calibration/source/confirmation counts: 10/5/5/6 objects.
- Split: domain-separated SHA-256 order of archive-derived canonical object
  tokens, assigned before any phase label is decoded.
- Probe budgets: 0, 1, 2, and 3 observations after the mandatory anchor.
- Primary contrast: object-level regret AUC for decision-directed selection
  minus the same AUC for task-agnostic latent-response information gain.
- Utilities: +1 for a stable chosen pose, -1 for an unstable chosen pose, and 0
  for abstention.
- Guard: one shared 80% object-level lower-stability certificate calibrated
  without confirmation objects.
- Confirmation: one attempt, exact paired object-level sign-flip test, no
  replacement or outcome-based method change.

The modest confirmation count makes this a demanding test: a statistical pass
requires a highly consistent object-level benefit. Nested grasps increase
precision within an object but never inflate the primary sample size.

## Current acquisition state

The public repository is locked to revision
`5e290eb024f25b1f4aa602724e6869e512aca434`. The primary data locator is the
official `gelsight.zip` Google Drive file. At protocol freeze, Google Drive
exposed the public locator but refused byte acquisition due to its download
quota. No archive member name, phase label, sensor payload, or shake outcome has
been opened.

## Preaccess mapping constraints

The official paper fixes the scientifically consequential feature boundary
without requiring archive access: prediction uses the grasp and pose phases,
twenty evenly spaced pre-shake timesteps, RGB, GelSight tactile, raw wrist
force/torque, and fixed gripper force. Tactile frames use pre-contact image
subtraction before a frozen ImageNet-pretrained ResNet50 feature extractor. The
shake phase supplies the later target and is never an input.

Those author-defined constraints and the experiment's fit-only preprocessing
rules are frozen in
`protocols/poseit_real_decision_probe_v1_preaccess_mapping_constraints.json`.
Its file SHA-256 is
`8bf66c087437d77589d5fcd35d74a47b2a4d8ba69b311041123d719da8445210`.
The clarification binds the original preregistration rather than modifying it.
Only mechanical facts that cannot be known before central-directory inspection
remain deferred: member path templates, released timestamp and phase-field
names, cadence-aware deterministic resampling, structural missingness, and the
resulting tensor layout. They must be frozen before any member payload is
opened; labels and outcomes cannot inform them.

## Shared selector kernel

`poseit_real_decision_selectors.py` implements the target-blind policy core over
one joint Gaussian belief containing per-pose pre-shake features and a separate
latent shake-stability coordinate. Conditioning can accept only a pose's
pre-shake feature vector; no selector or policy-trace function accepts a shake
label or outcome. The decision-directed objective integrates the expected
increase in best downstream Bayes utility, including zero-utility abstention,
using the registered 4,096 antithetic predictive draws. The task-agnostic
control instead maximizes Gaussian information gain about the same complete
latent stability vector. Both use the same posterior state and lowest-pose tie
break.

Synthetic contract tests demonstrate that the objectives can select different
probes and that structurally unavailable actions are excluded without dropping
an otherwise valid family. This is implementation evidence only. The joint
belief cannot be fitted, calibrated, or scientifically evaluated until the
archive-specific feature mapping has passed its earlier gates.

## Source-independent method lock

The remaining fit and analysis choices are frozen in
`protocols/poseit_real_decision_probe_v1_method_lock.json`, before archive byte
acquisition. Its file SHA-256 is
`4fa1ef3c96df28a67e13461b79c44690f53f5abb4c90e06200c4e90bcf8e1a1c`.
It binds the parent protocol, preaccess mapping constraints, and selector-kernel
revision.

The exact method uses fit-only coordinate standardization and an
outcome-independent, sign-canonical eight-component PCA over all structurally
present fit-pose records. Complete fit families form action-major rows in a
joint Gaussian twin: eight projected pre-shake coordinates and one latent shake
stability coordinate per pose. `Pass` maps to +1 and all unstable labels map to
-1. The empirical covariance uses 25% diagonal shrinkage and fixed relative
jitter. If a fit family is not structurally complete across all 16 poses, the
archive gate must fail before an outcome payload is opened.

Five calibration objects supply one shared simultaneous certificate. For each
object, the score is the maximum positive standardized latent-response
shortfall over every valid-anchor family, selector, budget, and available
action. The 80% finite-sample rank is the fifth of five object scores. Subtracting
that multiplier in standardized latent space yields a lower stability
probability through the standard-normal CDF. A policy takes the lexicographically
first pose whose lower bound reaches 0.5 and otherwise abstains. The same
certificate is used by both adaptive selectors and all controls.

The random-order control is also no longer underspecified. It is the exact mean
over 256 full pose orders derived by the locked SHA-256 rule; the order-roster
digest is
`889f81c2ec6b1f33e3f55e7a2d9e6f4e879b9bf511ec8a5ead9933d45fc9bee3`.
Family regret is integrated over budgets 0--3, then averaged within object
before inference. False-safe rate is conditional on taking a certified action;
unsafe-action rate uses every family-budget decision, including abstentions.
Object-level coverage requires simultaneous coverage across every family,
selector, budget, and available action.

`poseit_real_decision_analysis.py` implements this fit, calibration, evaluation,
source gate, and one-shot confirmation analysis. Its tests use synthetic
families only. They do not fit PoseIt, inspect an archive member, or constitute
scientific evidence.

The repository contains inconsistent license signals: its license file is CC0
1.0, while its README displays CC BY-SA 4.0 and MIT badges. This does not prevent
an attributed academic analysis, but raw archive bytes must not be redistributed
until the release terms are clarified.

## Gates

Before any scientific execution, the acquisition stage must:

1. acquire the exact public primary archive and publish its byte size and
   SHA-256;
2. inspect archive structure without decoding labels or sensor values;
3. confirm 26 canonical objects and freeze their hash-derived roles;
4. bind exact paths, pre-shake timestamps, resampling, missingness handling, and
   the fixed-dimensional feature map;
5. seal the implementation and pass independent source-only tests.

Only then may fit and calibration outcomes open. Source-test outcomes open once,
after predictions are sealed. Confirmation remains unauthorized until the
registered source gate passes.

## Range-hash custody for the public archive

The official Drive metadata and an exact HTTP 206 probe establish that
`gelsight.zip` is 905,738,058,282 bytes, was last modified at
`Sat, 20 Aug 2022 02:26:04 GMT`, and supports exact byte ranges even while a
full HTTP 200 download is quota-blocked. These transport facts do not expose a
ZIP member name or payload.

The source-independent transport contract is frozen in
`protocols/poseit_real_decision_probe_v1_range_transport_lock.json`, with file
SHA-256
`8b3843bd4255aae980e3c8474f60fb38431bdb61e043a9a2e062d1c2acf8b67a`.
It binds the exact range implementation at commit
`098d0090110fc321818ce22a5900675b3cc62632`. The registered execution hashes
every byte in ascending order using 32 MiB chunks and at most eight concurrent
requests. Bytes are released after entering the ordered SHA-256 state; no local
archive is retained. Identity failures are terminal, while each identical range
may receive at most three transport-only attempts.

The exact command on `gpuserver4090` is:

```bash
flock -n /home/florianpfaff/source-only/poseit-real-decision-v1/range-hash.lock \
  env PYTHONPATH=src python \
  scripts/science/acquire_poseit_gelsight_range_hash_v1.py \
  --receipt /home/florianpfaff/source-only/poseit-real-decision-v1/range-hash/acquisition-receipt-v1.json \
  --progress /home/florianpfaff/source-only/poseit-real-decision-v1/range-hash/progress-v1.json \
  --protocol protocols/poseit_real_decision_probe_v1.json \
  --expected-protocol-sha256 221803b109a82d3a2d923d5e0c18284b965a8848bcd69e25addd97409d31c5d4 \
  --mapping-constraints protocols/poseit_real_decision_probe_v1_preaccess_mapping_constraints.json \
  --expected-mapping-constraints-sha256 8bf66c087437d77589d5fcd35d74a47b2a4d8ba69b311041123d719da8445210 \
  --method-lock protocols/poseit_real_decision_probe_v1_method_lock.json \
  --expected-method-lock-sha256 4fa1ef3c96df28a67e13461b79c44690f53f5abb4c90e06200c4e90bcf8e1a1c \
  --transport-lock protocols/poseit_real_decision_probe_v1_range_transport_lock.json \
  --expected-transport-lock-sha256 8b3843bd4255aae980e3c8474f60fb38431bdb61e043a9a2e062d1c2acf8b67a \
  --range-transport-core src/bayesian_phystwin_experiments/poseit_remote_archive.py
```

The mutable progress file is monitoring metadata, not evidence. Only the
write-once completion receipt supplies the archive SHA-256 required by the
parent protocol. Central-directory ranges remain closed until that receipt
exists. Member payload ranges remain closed until the later structure and
archive-specific mapping locks authorize them.

### Retained pre-receipt transport failure

On 2026-09-02 UTC, the first range-hash process was observed absent, with no
completion receipt. Its retained traceback ends at the strict HTTP 206 guard.
The actual response status and cause were not recorded, so neither a quota
failure nor changed archive identity is inferred. The last persisted progress
was 7,945 of 26,994 chunks (266,589,962,240 bytes). Its attempt count covers
completed chunks, not all potentially issued concurrent requests. This prefix
is not a full archive hash and cannot authorize structure or scientific access.

The unchanged log and progress snapshot are preserved in
`evidence/poseit-real-decision-v1/range-hash-v1-failure/`, together with a compact
terminal observation. A subsequent single-response, header-only GET for byte
0 passed the unchanged identity validator; its response body was not read.

The already-frozen transport lock permits a replacement process from byte zero
after preserving a pre-receipt failure. The exact v2 replacement command and
parent evidence are bound in
`protocols/poseit_real_decision_probe_v1_range_restart_v2.json`. It uses the same
source deployment, lock, chunk sizes, retries, worker count, and full-archive
receipt path, with distinct progress/log paths. It does not overwrite the first
attempt, resume from an unverified prefix, or change any scientific choice.

The replacement was launched once at `2026-09-02T16:57:14.387945+00:00` under
flock PID `3994831`; an independent process check confirmed Python child
`3994832` and an advancing `range-hash/progress-v2.json`. The write-once attempt
ledger and launch record are retained in
`evidence/poseit-real-decision-v1/range-hash-v2-launch/`. They prove a launch,
not archive completion, and do not authorize re-execution when a later poll
times out. The original `progress-v1.json` and `run-v1.log` stay untouched.

The replacement subsequently terminated at the same HTTP 206 guard after 87
chunks (2,919,235,584 bytes), without a receipt. Both PIDs were independently
confirmed absent. Its immutable snapshots are retained in
`evidence/poseit-real-decision-v1/range-hash-v2-failure/`. A separate header-only
GET for the next unfinished range returned HTTP 200 with a 2,009-byte HTML
content type, no content range, and no archive identity headers. The body was
not read; this diagnostic does not establish the cause of the earlier response.
The byte-zero header check alone therefore did not establish reliable delivery
of the complete archive.

The v2 attempt record is consumed and authorizes no second process. No hash
process is currently running, and no third full-hash attempt was launched.
The next prerequisite is stable delivery of the same official archive, such as
a complete attributed author-distributed copy. Any further transport execution
must preserve both failures and be frozen before launch. The source/method locks
and full-hash-before-structure requirement are unchanged; no scientific claim
follows from the implementation tests or these transport observations.

### Quota diagnosis and prospective checkpoint recovery

A later bounded transport diagnostic at
`2026-09-02T17:08:20.674819+00:00` read only the 2,009-byte HTML error response,
not archive content. It explicitly identified a Google Drive download-quota
error. This new observation does not retroactively establish the exact contents
of either failed run's response. The administrative recovery path now includes
no further provider request before `2026-09-03T17:08:20.674819+00:00`; that is a
conservative 24-hour pause, not a known reset time or permission to start a job.

The checked server, local Linux, and Windows filesystems have less free space
than the complete archive requires. Nothing was deleted or reallocated.
`evidence/poseit-real-decision-v1/delivery-recovery-feasibility-v1.json` records
these source-independent observations and a local synthetic prototype.

`poseit_hash_checkpoint.py` explores durable hashing without retaining the
archive or inventing a new SHA-256 implementation. It uses the upstream
[RHash v1.4.6 export/import API](https://github.com/rhash/RHash/blob/v1.4.6/librhash/rhash.h),
pinned to commit `6562de382954d9893442b89b0e8b5c513eea6a88`. Only the exact
non-OpenSSL native ABI and a caller-supplied binary SHA-256 are accepted.
Checkpointing is restricted to complete 64-byte blocks. The native input
buffer must be zero; irrelevant structure padding is zeroed before storage.
A final partial block is processed only transiently on a cloned context.
Restoration checks the library binding, external checkpoint ID, native layout,
byte counters, padding, and prefix digest before a state is returned.

The 132-byte native state contains a derived hash state and counters, not an
input fragment. Forty-seven local synthetic tests include Python `hashlib`
agreement, multiple restarts, deterministic partitioning, the registered 32 MiB
chunk size, a restart across 4 GiB, and corrupted metadata/native-state rejection.
The complete PoseIt suite passes 149 tests with the explicit test library.
These native tests are optional and skipped when that library is not provided;
there is no system-library fallback. The upstream shared-library tests and
strict MyPy for the new module also pass. No remote library was changed.

This is not yet an acquisition runner or a transport amendment. Before it could
be used on PoseIt, the next implementation must add atomic, hash-chained,
write-once checkpoint/attempt custody; reject rollback, missing/reordered ranges,
or a changed native library; preserve quota failures and the provider cooldown;
and emit the existing full-archive proof only after every byte was hashed in
order. A separately frozen transport amendment must bind those changes before
one new acquisition is launched. The prior failed prefixes cannot be recovered,
so that new attempt would still start at byte zero. Scientific choices and the
full-hash-before-structure gate must remain unchanged throughout.

To reproduce the synthetic native tests with the isolated source checkout:

```bash
git clone --depth 1 --branch v1.4.6 https://github.com/rhash/RHash.git /tmp/poseit-rhash-v1.4.6
cd /tmp/poseit-rhash-v1.4.6
git rev-parse HEAD
./configure --disable-openssl --enable-lib-shared --disable-lib-static
make -j2 lib-shared
make -j2 test-lib-shared
```

Then, from the experiment worktree:

```bash
env PYTHONPATH=src \
  POSEIT_TEST_RHASH_LIBRARY=/tmp/poseit-rhash-v1.4.6/librhash/librhash.so.1.4.6 \
  uv run --no-project --python 3.12 --with pytest --with numpy --with scipy \
  python -m pytest --capture=sys -q tests/test_poseit_*.py
```

## Full-download and structure-only custody tools

If a complete local copy becomes available, the alternate exact acquisition
command is:

```bash
PYTHONPATH=src python \
  scripts/science/acquire_poseit_gelsight_archive_v1.py \
  --archive /home/fpfaff/source-only/poseit-real-decision-v1/archives/gelsight.zip \
  --receipt /home/fpfaff/source-only/poseit-real-decision-v1/archives/acquisition-receipt-v1.json \
  --protocol protocols/poseit_real_decision_probe_v1.json \
  --expected-protocol-sha256 221803b109a82d3a2d923d5e0c18284b965a8848bcd69e25addd97409d31c5d4 \
  --mapping-constraints protocols/poseit_real_decision_probe_v1_preaccess_mapping_constraints.json \
  --expected-mapping-constraints-sha256 8bf66c087437d77589d5fcd35d74a47b2a4d8ba69b311041123d719da8445210 \
  --method-lock protocols/poseit_real_decision_probe_v1_method_lock.json \
  --expected-method-lock-sha256 4fa1ef3c96df28a67e13461b79c44690f53f5abb4c90e06200c4e90bcf8e1a1c
```

The acquisition tool accepts only the frozen official Google Drive file ID and
the registered `gelsight.zip` name. It rejects HTML, quota pages, renamed files,
non-HTTPS redirects, and unregistered redirect hosts. A successful response is
streamed opaquely into a write-once destination while computing SHA-256. The
content-bound receipt records that no ZIP structure, member name, sensor value,
phase label, or shake outcome was opened.

After successful retained acquisition, the first admissible local
archive-inspection command is:

```bash
PYTHONPATH=src python \
  scripts/science/build_poseit_archive_structure_lock_v1.py \
  --archive /home/fpfaff/source-only/poseit-real-decision-v1/archives/gelsight.zip \
  --protocol protocols/poseit_real_decision_probe_v1.json \
  --expected-protocol-sha256 221803b109a82d3a2d923d5e0c18284b965a8848bcd69e25addd97409d31c5d4 \
  --mapping-constraints protocols/poseit_real_decision_probe_v1_preaccess_mapping_constraints.json \
  --expected-mapping-constraints-sha256 8bf66c087437d77589d5fcd35d74a47b2a4d8ba69b311041123d719da8445210 \
  --method-lock protocols/poseit_real_decision_probe_v1_method_lock.json \
  --expected-method-lock-sha256 4fa1ef3c96df28a67e13461b79c44690f53f5abb4c90e06200c4e90bcf8e1a1c \
  --private-member-manifest /home/fpfaff/source-only/poseit-real-decision-v1/structure/private-member-manifest-v1.json \
  --output /home/fpfaff/source-only/poseit-real-decision-v1/structure/archive-structure-lock-v1.json
```

The tool hashes the complete ZIP and reads its central directory. It rejects
encrypted, duplicate, traversing, absolute, linked, and special members. It does
not call `ZipFile.open`, decompress a member, verify member payload CRCs, decode a
phase label, or read sensor data. Member names stay in the local private manifest;
the compact lock contains only aggregate structure and domain-separated digests.
