# Deform360 Adaptive Causal Direct Depth V14

## Status

V14 is implementation-locked before selecting a fresh source cohort. It is a
prospective source-development study, not a result, confirmation, or
state-of-the-art claim. Only a complete pass of the frozen source gates may
authorize a separately preregistered evaluation on independent fresh objects.

## Why This Is Not Another Tracker Arm

V13 established two separate facts. Its target-free camera carrier succeeded
on six of eight opened source cases, but its fixed frame-zero TAPNext++
identity provider retained only 5 of 96 endpoint identities. V14 keeps the
former and discards the latter.

The V14 carrier chooses eight complete cameras at frame zero and splits them
into disjoint four-camera proposal and validation panels. At every tested
causal endpoint, V14 re-associates local metric RGB-D geometry around the
selected physical or persistence backbone. It does not propagate a fixed
camera track identity through the prefix.

The strict arm requires three supporting views in each panel. The fallback arm
requires two views in each panel, multiplies local covariance by four, and
retains a separate 5 mm shared-camera-bias nuisance. Cases without either
carrier abstain before source locking.

## Causal Boundary

V14 scans only frames 0 through 57 and selects the earliest endpoint that
passes all registered conditions:

1. the released tactile stream supports contact;
2. the measured actuator has moved by the required amount;
3. observed nonrigid response is large enough;
4. the response aligns with the physical action-conditioning trajectory;
5. proposal and validation panels agree under the shared-bias covariance; and
6. the proposal correction improves the untouched validation-prefix residual.

The proposal panel compares physical motion with exact persistence. Physical
motion must win by 5% to become the prediction backbone; otherwise persistence
is selected. The physical trajectory remains a separate action-conditioning
signal in either case.

The proposal panel forms the candidate. The validation panel can only admit or
reject it. Every rejection returns the selected baseline byte for byte. No
object observation after frame 57, hidden identity, future geometry, or future
metric may be read before all twelve source predictions or exact fallbacks are
sealed.

## Reliability And Calibration Boundary

Association probability comes from local candidate geometry and mask/depth
support. Prior reliability comes from residual-independent redundancy and
view-scatter cues. The innovation against the selected twin trajectory is not
fed back into either prior quantity; it enters the robust mixture likelihood
once.

Metric covariance is carried in square metres. It includes local depth and
pixel uncertainty, assignment-mixture spread, temporal unknown-correlation
inflation, arm-specific two-view inflation, cross-view scatter, bounded
per-endpoint Sim(3) fit residual, and the shared-camera-bias variance.

The per-endpoint Sim(3) operation is a nuisance debiasing step, not evidence
that global scale or pose is known. It removes bounded global pose and scale
from the sparse endpoint displacement before the nonrigid readout discrepancy
is inferred.

The resulting candidate remains an observable/readout correction. V14 does
not claim to update the hidden Warp state or to identify a physical material
mechanism.

## Freshness And Admission

The source cohort will contain twelve physical objects, one episode per
object, split into three object-level cross-fit folds of four. The hash-only
exclusion union contains 138 physical objects previously opened, reserved,
selected, or technically dispositioned by relevant Bayesian-PhysTwin,
Prob4D, MolmoMotion-Field, and held-v8 work.

Before a case can enter the source lock, an outcome-blind preflight must verify:

- released metadata enums;
- 76-frame episode, robot, and tactile streams;
- the physical backend's 128-to-10,000-node contract;
- all twelve registered camera records;
- at least eight complete depth, mask, and calibration streams;
- complete source checksums; and
- an admitted strict or inflated adaptive carrier.

The preflight retains hashes rather than plaintext object or episode identity.
A rejected preflight is not a selected source case. Once selected, a case is
never silently replaced after any prediction or outcome disposition.

The metadata-only staging queue was sealed before episode media were listed or
downloaded. After applying the 138-object exclusion, the public catalog
contained 54 fresh objects. One malformed metadata record was rejected
hash-only for an invalid `bimanual` enum, leaving 53 ordered candidates. The
first twelve queue positions contain four sheet, four compact, and four complex
objects; all remaining candidates are immutable preflight reserves.

The queue is
`configs/sota/deform360_causal_response_direct_depth_v14_staging_queue.json`,
with internal digest
`e56dd42c07ce4aff0e3e0e2913e00a1e80d0727beef1884b1de93198f32bec75`
and file SHA-256
`225ff37dd8ec2ff1ae12340ebfa3dc27fb6b045edee95d3b675832a72fa43d74`.
The source panel is the first twelve accepted outcome-blind V14 preflights in
this order. Predictions, metrics, and outcomes cannot trigger replacement.

For each attempted rank, the indexed downloader admits only the exact episode
recordings for the twelve registered cameras, refined calibration, metadata,
and exact-stem tactile arrays and timestamps. It selects the latest tactile
baseline recorded no later than the episode, falling back to the earliest
available baseline only when no preceding baseline exists. Audio and unrelated
episodes remain unstaged. Robot motion is recovered from the registered camera
panel by the pinned official Deform360 robot stage; it is not inferred from
future object motion.

Before any object response is used, the pinned action-only selector chooses an
81-frame processing window from the complete released robot trajectory. The
same half-open frame interval is then applied to RGB, robot, and synchronized
tactile streams. The last five processing frames are reserved for the official
point-cloud tail policy, leaving the registered 76-frame predictive episode.
Full known action is permitted for window selection; object RGB and tactile
values are not.

The prefix-asset amendment at
`configs/sota/deform360_causal_response_direct_depth_v14_assets.json` corrects
an implementation mismatch found before source object decoding. The original
preflight helper compared camera mask and depth assets to the 76-frame
prediction trajectory, despite the method boundary permitting object
observations only through frame 57. Prediction-facing RGB, masks, and depth
are now exactly 58 frames; robot, tactile, and the physical prediction remain
76 frames. No future camera asset is created before all source predictions or
exact fallbacks are sealed. This changes neither a method threshold nor an
advancement gate.

Ranks 1 and 2 are preserved as pre-lock technical staging failures. Ranks 3
through 14 produced complete 81-frame camera, robot, and tactile windows. They
remain ordered preflight candidates rather than selected source cases; the
source panel is still the first twelve accepted outcome-blind preflights in
the immutable queue.

The prefix mask stage is now complete for ranks 3 through 14. All twelve
candidate artifacts are ready for geometry, with 140 of 144 camera streams
successful. Failed camera streams remain explicit dispositions and are not
silently substituted.

Prefix geometry has its own immutable child lock at
`configs/sota/deform360_causal_response_direct_depth_v14_prefix_geometry.json`.
It binds the exact twelve mask artifacts, official Deform360 source files,
CUDA/gsplat runtime, reconstruction settings, and deterministic frame-zero
point seeding. Reconstruction consumes only frames 0 through 57. It records
58-frame RGB, mask, rendered-depth, and gripper-mask counts per retained
camera, validates calibration, and writes a hash-only frame-zero geometry
manifest.

The physical point-count contract is enforced before source locking:
`start_obj_pcd.ply` must contain between 128 and 10,000 finite nodes. A
geometry or backend-admissibility failure is a pre-lock technical failure, not
a model prediction. The number of seeded nodes supported by frame-zero metric
depth is recorded as a target-free diagnostic but is not an additional
admission threshold. No tracker, future point-cloud trajectory, hidden
identity, or outcome is available to this stage.

The first server smoke stopped before prefix materialization or reconstruction
because importing the pinned gsplat 1.4.0 backend rebuilt its JIT extension.
The package, source, Python, Torch, CUDA, compiler, path, and backend probe were
unchanged, but the compiled binary checksum changed. No geometry or result
artifact existed, so the candidate remained pre-lock and unattempted.

The separate runtime amendment at
`configs/sota/deform360_causal_response_direct_depth_v14_prefix_geometry_runtime.json`
records that failure and pins the exact rebuilt binary. It changes no geometry
setting, method threshold, source ordering, or advancement gate. Every later
geometry result must carry a checksummed runtime-application sidecar; the
parent geometry protocol and builder remain unchanged.

The first completed geometry smoke then exposed a validation-only filename
assumption: official Deform360 writes the frame-zero splat as `splat_0.ply`,
while the original post-run validator looked for `splat_000000.ply`. The
manifest already bound the hash of the actual official file, and all other
fixed outputs matched. The separate validation amendment at
`configs/sota/deform360_causal_response_direct_depth_v14_prefix_geometry_validation.json`
therefore validates the official filename without rewriting the successful
manifest, geometry result, or runtime sidecar. It changes no reconstruction
byte, method threshold, queue disposition, or advancement gate.

A later two-GPU operator invocation first selected system Python, which lacks
OpenCV, and then selected the exact Deform360 virtual environment. Both ranks
stopped before the parent builder materialized prefix inputs. Loading gsplat
from the correct interpreter relinked the same JIT sources and changed only
the compiled extension checksum. The relinked binary was then stable across
serial and concurrent imports. The second runtime-only amendment at
`configs/sota/deform360_causal_response_direct_depth_v14_prefix_geometry_runtime_v2.json`
binds the exact interpreter and relinked extension for ranks 4 through 14.
Rank 3 remains bound to the first runtime amendment; no existing artifact is
rewritten, and the two failed invocations are not case attempts.

The first runtime-v2 smoke completed successfully with twelve cameras and 856
physical nodes. A second non-mutating validation amendment binds its manifest,
result, runtime-v2 sidecar, official `splat_0.ply` output, and the complete
runtime-amendment chain before the remaining ranks are permitted to run.

## Advancement Gate

The source study must seal twelve predictions or exact fallbacks without a
technical failure. At least six objects must admit an update. Relative to the
unchanged selected baseline, V14 must:

- improve object-balanced disjoint hidden-identity error by at least 5%;
- improve object-balanced Chamfer distance by at least 5%;
- jointly win on at least 8 of 12 objects;
- keep every single-object regression below 5%;
- keep false-safe admissions below 10%; and
- pass the registered three-fold source-calibrated upper-regret guard.

Early, middle, and late errors, NEES, coverage, and interval width are reported
regardless of the decision. Failure closes V14 without threshold changes.
Passing authorizes only a new independent target protocol; it is not itself a
confirmation.

## Synthetic Controls

Before source locking, the production V14 wrapper must pass frozen strict and
inflated positive controls plus rigid-bias, cross-panel-inconsistency, and
missing-contact placebos. These controls test implementation sensitivity,
specificity, covariance routing, and exact fallback. They are not real-data
evidence.

The frozen controls passed: 12 of 12 positive trials produced an update, none
of 12 placebos was admitted, all 12 placebos preserved the baseline exactly,
and mean synthetic continuation error improved by 11.64%. The source-lock
builder validates and binds this checksummed result; it cannot create a cohort
lock from a missing, altered, or failed control artifact.

## Claim Boundary

The strongest possible source-stage statement is:

> On twelve fresh development objects, a prospectively locked,
> tactile-supported adaptive direct-depth readout update passed or failed its
> registered transfer, safety, and calibration gates.

No SOTA language is permitted until a later independent target protocol is
locked and completed.
