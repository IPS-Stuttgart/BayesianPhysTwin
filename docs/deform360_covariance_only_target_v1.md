# Deform360 covariance-only target v1

## Purpose

The opened 22-case evidence supports a narrow Bayesian claim: endpoint-model
covariance can improve a proper predictive-distribution score even when no new
point mean beats exact last-residual persistence. This protocol tests that claim
on previously untouched Deform360 physical objects.

The candidate changes covariance only. Its point mean is byte-identical to
`last_residual`; its covariance is donated by `independent_endpoint_v1` and
multiplied by `8`, `16`, and `16` in the early, middle, and late horizons. These
numbers multiply covariance, not standard deviation. If no causal residual is
available, the unchanged physical prediction remains the exact fallback.

## Target selection

All previously opened, reserved, selected, or technically dispositioned objects
are represented by a hash-only exclusion artifact. From the remaining official
object names, the protocol fixes 16 candidates per object stratum before reading
metadata. Only `metadata.json` may then be opened.

The final target contains 24 distinct physical objects and one session per
object. It has exactly two sessions in every cell of:

- sheet versus volumetric object naming stratum;
- unimanual versus bimanual interaction; and
- elevation, planar/contact, versus shape-change action family.

If the fixed candidate panel cannot satisfy the factorial design, the protocol
stops before target payload access. Once the roster is frozen, missing streams,
missing released annotations, reconstruction failures, and unsupported sessions
remain in the denominator and are never replaced.

## Information boundary

Before roster lock, code may read only official object names, selected
`metadata.json` files, repository filenames/sizes/digests, and hash-only prior
exclusions. It may not decode camera media, load tactile or robot arrays, open
geometry or track annotations, run reconstruction or tracking, visualize a
target, or inspect any future score.

After roster lock, exact payload bytes may be downloaded and rehashed into a
quarantined cache. Released Deform360 annotations are preferred where they exist;
their absence is a retained target disposition, never a reason to substitute a
different object.

## Evaluation

The primary endpoint is the equal-object mean Gaussian negative-log-score
difference between the covariance-only candidate and `last_residual`, using the
registered common 5 mm observation-noise floor. Physical object sessions are the
independent resampling units. Coverage, interval width, NEES, energy score, and
horizon-resolved behavior are secondary outcomes. Track and Chamfer predictions
must remain exactly identical because the mean is unchanged.

This protocol cannot support a point-accuracy, physical-state, Causal4D, Prob4D,
or state-of-the-art claim. A paper claim requires the independently locked target
gate to pass with complete 24-session failure accounting.

## Metadata-gate results and v1.2 amendment

The fixed 32-object v1 candidate panel was opened at the metadata-only boundary.
Strict validation then stopped because one released episode has malformed
`nonprehensile` metadata. No target roster was created and no camera, tactile,
robot, geometry, track, or outcome payload was opened. The complete panel was
retained in a hash-only exclusion artifact; no replacement occurred.

A names-only capacity audit found that discarding the panel would leave only two
untouched non-`-cloth` objects, making a new balanced 24-session panel impossible.
Protocol v1.1 therefore reuses the identical panel and makes one schema-only
change: `nonprehensile` is recorded with a validity flag but cannot affect
eligibility or assignment. `action` and `bimanual`, the two metadata fields that
actually define the factorial cells, remain strict. The amendment was frozen
before roster creation or target payload access.

The unchanged v1.1 panel then stopped a second time because the released action
`pull short side` begins with `pull`, which was absent from the coarse action
registry. Still before roster creation or payload access, one complete domain
audit was run over the 320 already-open episode records. It found 17 distinct
first tokens, clean `yes`/`no` bimanual values, three malformed record-only
`nonprehensile` values, and no structural metadata errors. The only action tokens
missing from the registry were `pull`, `open`, and `close`.

Protocol v1.2 therefore makes one finite vocabulary-only amendment: `pull` joins
the planar/contact family, while `open` and `close` join the shape-change family.
The identical 32-object panel, metadata bytes, seed, quotas, solver, method,
fallback, covariance scales, endpoints, and gates are retained. This amendment
was committed before creating a target roster and before opening any target
payload or outcome.

## Frozen target roster

The v1.2 metadata-only gate succeeded on the unchanged candidate panel. The
frozen roster contains 24 unique physical objects and exactly two sessions in
each of the 12 registered object-stratum, bimanual, and action-family cells. Its
roster SHA-256 is
`f9106b3dd6e0cec089623e07fed3506755fb334952c7761846d0854dfba45783`.

The roster and full metadata-only selection are bound by
`results/science/deform360_covariance_only_target_v1/target_roster_v1_2.json`.
At roster lock, no target media, sensor array, geometry, track annotation,
prediction, or outcome had been opened. The next permitted operation is an
exact-file availability plan and quarantined byte download for these 24 sessions;
no target may be replaced.

## Frozen exact-file plan

The full roster names-only audit retained all 24 sessions. Twenty satisfy the
ordinary raw support contract. Four remain in the denominator as technical
support failures: three have ambiguous nearest tactile baselines in all four
tactile streams, and one has tactile baselines that do not form one cross-sensor
capture. Camera availability is strong in every row (37--41 streams), so the
failure classification is specifically tactile-association provenance rather
than missing RGB support.

No selected object has exact official processed annotations at the pinned
Deform360 revision. The sealed plan contains 2,154 exact raw files totaling
2,464,053,620 declared bytes, with plan SHA-256
`d5bab5a05cf49ba6cc7bd31ffe57d2abc15040dd3f2de163d5f5034800b3ee51`.
It is stored at
`results/science/deform360_covariance_only_target_v1/exact_file_plan_v1.json`.
No payload byte was opened while building the plan.

## Quarantined download

All 2,154 planned files were downloaded and independently rehashed, totaling
2,464,053,620 bytes. Every declared LFS digest also matched. The download
manifest SHA-256 is
`41bfb0feb246ac235e6364cfb46304dd8b2679801d73532a1e78281f243d59af`;
the independent verification SHA-256 is
`3f58caf2e5cff977b34ddce6f42c86438696ce9f36f637238597f4ca86c15997`.
The payload tree is read-only under the registered quarantine root. Downloading
and hashing did not decode media, load arrays, run predictions, or open outcomes.

Because none of the 24 objects has official processed annotations, this cohort
is a custom fresh-object calibration test, not an official Deform360 benchmark
parity claim. Before any decode or prediction, a separate source-only provider
gate must prove: metres/metres-squared units; coordinate-frame, material-identity,
and horizon alignment; a causal residual history of shape `(T,N,3)` with `(T,N)`
validity; PSD covariance; byte-identical `last_residual` means; and exact fallback
on provider failure. Missing prefix frames remain missing rather than being
nearest-filled.

The eventual outcome must use cameras and processing artifacts disjoint from the
prefix residual provider. In particular, donor and target may not share one
reconstruction artifact; otherwise a common-mode camera error could calibrate
against itself.

## V1.3 provider and custom-evaluation lock

Protocol v1.3 freezes the executable causal residual-history adapter and the
custom evaluation before any target decode. It leaves the v1.2 roster, method
mean, covariance donor, scales, fallback, and no-replacement rule unchanged.

The adapter preserves a complete `(T,N,3)` causal history and `(T,N)` validity;
missing rows are never nearest-filled as observations. Identities need two valid
updates, and a case needs two observed frames plus 50% empirically supported
identities. Prior-only covariance is not accepted as evidence. Unsupported
identities and failed cases use exact fallback.

Because official processed annotations are absent, the official track/Chamfer
checks are marked unavailable. Registered scores are per-identity 3D marginal
NLL, NEES, coverage, and interval volume, aggregated within session and then
equally across sessions. Provider and scoring cameras use a deterministic
names-only hash partition and distinct reconstruction artifacts. The source-only
synthetic gate passed without target access; details are in
`docs/deform360_covariance_only_target_v1_3.md`.

## V1.4 pre-decode provider correction

Independent source-only review reproduced two blockers in the v1.3 provider
before any target decode. Protocol v1.4 fixes admissible geometry association,
candidate-specific innovations, assignment-mixture covariance,
residual-independent cue reliability, canonical unique windows, and explicit
observation-split/reference-mean digest binding. Distant rows cannot manufacture
support. A provider-specific heteroscedastic robust endpoint update consumes the
metric covariance and cue reliability once while preserving the registered mean.

The v1.2 roster, acquisition artifacts, method mean, covariance donor, scales,
support thresholds, fallback, and evaluation remain unchanged. The v1.3 lock is
preserved and superseded prospectively. Details are in
`docs/deform360_covariance_only_target_v1_4.md`.
