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

That initial prototype did not provide an acquisition runner or authorize a
transport amendment. The checkpointed runner described below is a subsequent
software-only development. It does not recover either prior failed prefix,
so a future initial checkpointed attempt must still start at byte zero.
Scientific choices and the full-hash-before-structure gate are unchanged.

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

### Checkpointed transport implementation

`poseit_checkpoint_acquisition.py` now implements ordered range hashing with
immutable per-range native checkpoints, content-bound attempt authorizations,
start and terminal records, and a separate completion receipt. Every range
still passes the original exact HTTP 206 and archive-identity validator before
its body is read. The wrapper also treats an HTTP error raised by Python's
opener as a terminal rejected response, not as a retryable socket failure.
Only the unchanged bounded socket/transport retries remain inside an attempt.
Once a worker fails, new provider requests stop and already-inflight workers
are joined before a failure is published.

A new attempt must provide an externally pinned authorization for the exact
previous terminal record, checkpoint count and tip, and an elapsed cooldown.
The shared external flock and write-once attempt directory prohibit duplicate
execution under the same authorization. Resume validation rejects altered,
missing, reordered, linked, or orphaned custody records. The new attempt starts
at the first uncommitted range; prior records are never overwritten. A missing
terminal record after an abrupt process loss is not permission to resume.

Only complete SHA blocks enter persisted native states. The final partial tail
is hashed transiently; no archive bytes or decoded contents are retained.
The acquisition producer computes the full archive SHA-256. Later offline
verification checks the stored chain and native checkpoint bindings; it does
not pretend to rehash unavailable raw bytes. A crash after complete terminal
custody but before receipt publication can be repaired without network access.
Unknown transport attempts for uncommitted concurrent work remain explicitly
unknown rather than being counted as zero.

`scripts/science/acquire_poseit_checkpointed_range_hash_v1.py` provides separate
`preflight`, `run`, `verify`, and `publish` modes. It requires the exact external
SHA-256 of a separately frozen transport amendment. That amendment must bind
the implementation files and revision, unchanged protocol and failure evidence,
exact imported source tree, isolated native library, host, output directory,
shared lock, and cooldown. There is no output-root override, implicit attempt
authorization, system-native-library fallback, or automatic restart. Preflight
loads only code and administrative metadata and grants no execution permission.

The new completion schema is deliberately distinct from the old range-hash
receipt. It is not silently accepted by the existing structure executor. An
explicit, reviewed structure-receipt integration is required before it could
authorize central-directory inspection, even after a full hash is available.

The tests use only synthetic byte streams and mocked command dispatch. They
exercise ordered hashing with out-of-order parallel responses, interruptions
and separately authorized resumptions, exact `hashlib` agreement, terminal
HTTP/identity failures without body reads, bounded socket retries, rollback and
authorization rejection, shared-lock exclusion, and offline receipt recovery.
They establish transport behavior, not the scientific decision-regret claim.

The final local PoseIt suite passed 230 tests with the explicitly selected
native library. An independently built isolated RHash library on
`gpuserver4090` (`workstation1`) has the identical binary SHA-256 and passed
upstream shared-library tests. The 128 native/checkpoint/command tests also
passed on that host using only a minimal code/protocol/administrative bundle
and synthetic streams. No system library was installed or replaced. The exact
file hashes, test observations, and remote build paths are recorded in
`evidence/poseit-real-decision-v1/checkpoint-transport-qualification-v1.json`.

### Frozen deployment and initial authorization

The implementation is sealed at
`8ba9f8668d66c887b94b86d30bc4075f6b6daae4`. A minimal 17-file code,
protocol, and administrative-evidence archive from that commit has SHA-256
`f207f7a7f9d12209a0021f37fce3038316920e145bb01157a9117862bdc3eeb0`.
It is deployed at
`/home/florianpfaff/source-only/poseit-real-decision-v1/source-checkpoint-8ba9f866`.
Both the archive hash and the exact deployed file roster were independently
rechecked on `gpuserver4090`; this is not a PoseIt data archive.

The separately committed transport amendment is
`protocols/poseit_real_decision_probe_v1_checkpoint_transport_amendment.json`,
SHA-256 `960a62785bce0b235644295f1c568f7bbdea462a3831cc53f673c72bbaccb43c`.
The unconsumed initial authorization is
`protocols/poseit_real_decision_probe_v1_checkpoint_attempt_000000.json`,
SHA-256 `8d523d7713ed3b3ba965ca415cdfc0eb869bfc0964580147d0a997159fc733e9`.
It binds attempt zero, an empty checkpoint prefix, and exactly one process to
the deployed specification. It cannot reuse either older failed prefix.

The actual server preflight and a separate read-only authorization/deployment
check are retained in
`evidence/poseit-real-decision-v1/checkpoint-deployment-v1/`. At
`2026-09-02T18:27:48.888832+00:00`, the code/native/protocol bindings passed,
the shared lock was available, the output root did not exist, the attempt count
was zero, and the authorization was unconsumed. Both observations explicitly
record no provider contact and no scientific authorization. Launch was not
permitted because the provider cooldown had not elapsed.

The exact future run and verification commands, log/launch custody, monitoring,
and failure policy are frozen in
`protocols/poseit_real_decision_probe_v1_checkpoint_execution_plan.json`.
The new CLI holds the shared flock internally; do not add an outer flock on
that same path. Do not overwrite or reuse a launch log, attempt, or checkpoint.

The next admissible steps are:

1. Wait until both current UTC and the server clock reach
   `2026-09-03T17:08:20.674819+00:00`. Recheck code/environment identity,
   process state, and unconsumed authorization before launching the frozen
   command once. This time is not a known provider reset and does not guarantee
   successful delivery.
2. Preserve any failure without automatic retry. On clean completion, verify the
   complete new-schema archive receipt and its custody chain.
3. Qualify and authorize the explicit structure-receipt integration, then pass
   the existing structure, mapping, source, and confirmation gates in order.

No new acquisition has been launched. No PoseIt scientific result exists.

### Checkpoint receipt to structure integration

`poseit_checkpoint_structure.py` and
`scripts/science/build_poseit_checkpoint_structure_v1.py` provide a separate
structure path for the checkpointed receipt. They do not modify the frozen
acquisition implementation, the original remote ZIP parser, or any scientific
lock. They do not reinterpret the new receipt as a legacy receipt.

Before a provider request, the new path requires a separately hash-pinned
`checkpoint-structure-authorization` binding the completed acquisition receipt,
archive digest, acquisition specification/amendment, exact implementation files,
host, output paths, structural size bounds, and cooldown. It uses the registered
acquisition verifier to recompute complete terminal/checkpoint custody under the
same shared flock. The structure attempt is consumed in a write-once ledger
outside its output directory before ZIP initialization. There is no alternate
output-root argument, implicit authorization, or automatic retry.

Each requested ZIP structure slice is served only after fetching and hashing its
covering acquisition chunks against their retained per-range SHA-256 values.
The HTTP identity checks remain unchanged. Whole chunks can contain adjacent
opaque compressed bytes; those bytes are held temporarily in bounded memory,
never decoded or written as a data payload. Only the requested end-record and
central-directory slices reach the original parser. The audit distinguishes
actual provider attempts from in-memory parser responses, and records no raw
bytes or member names in the public result. Exhausted transport retries, HTTP
rejection, and changed bytes are terminal; the parser cannot multiply retries.

The private member manifest, public structure lock, and successful terminal must
all exist for the new offline verification command to accept publication. That
command also rechecks acquisition custody and exact external authorization and
result hashes. It verifies retained records, not unavailable raw archive bytes.
Deleting an output directory does not release its external attempt ledger.

The implementation is sealed at
`9e4df0468f96346dd7b25583ba0053e0ce4ba8b6`. The complete local PoseIt suite
passed 275 tests with the explicit native library; without that library, 138
passed and 137 native-dependent tests skipped. An exact 22-file code/test bundle
from this commit was hash-verified on `gpuserver4090`; all 45 new structure tests
passed there. Ruff, strict MyPy for the module/command, and diff checks passed.
The compact software-only record is
`evidence/poseit-real-decision-v1/checkpoint-structure-qualification-v1.json`.

This qualification uses synthetic ZIPs, native checkpoint round trips, and
injected failures exclusively. No real structure authorization
has been issued, no real member name has been inspected, and no source or
confirmation access follows from these software tests. The next production
step remains the future-only checkpoint acquisition after its existing cooldown;
only a verified full archive receipt can permit the separate structure
authorization to be completed and reviewed.

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
