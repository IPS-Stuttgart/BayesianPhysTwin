# Prob4D observation timestamp consumption

Prob4D can attach a content-addressed timestamp-lineage sidecar to an existing
observation-factor bundle. BayesianPhysTwin consumes that sidecar without a
runtime dependency on Prob4D and independently revalidates its content identity,
source revision, exact factor order, frame indices, and bundle-manifest bytes.

This boundary exists because a frame index is not a physical timestamp. During
fast deformation, a camera-to-actuator or provider-to-simulator offset can look
like a coherent spatial discrepancy. Treating every timing effect as independent
point noise can make the update both physically wrong and overconfident.

## Two distinct timing uncertainties

The producer contract deliberately separates:

1. **Conditional factor-local jitter.** `conditional_timestamp_std_ns` excludes
   the shared clock offset. BayesianPhysTwin maps each selected observation row
   back to its exact source factor and forms one low-rank timing factor per
   recorded factor. Rows from the same factor therefore retain their common
   timestamp perturbation, while different factors are independent only under
   this declared representation.
2. **Coherent clock-domain offset.** One source/calibration-derived prior can be
   referenced by `shared_clock_offset_prior_artifact_id`. BayesianPhysTwin keeps
   this as an explicit timing state with design `dh/dt`; it is not inserted into
   local point covariance and is not duplicated in the conditional-jitter
   factors.

The corresponding observation model is

```text
y_i = h_i(x) + (dh_i/dt) delta_clock
      + (dh_i/dt) epsilon_factor(i) + measurement_noise_i,
```

where `delta_clock` is persistent within the named clock domain and each
`epsilon_factor` has the producer-supplied conditional scale.

## Claim-bearing loading and binding

Internal or exploratory code can use the portable loader directly. A promoted
or claim-bearing run must use the admission wrapper and supply both the raw
source digest and the content ID of the separate source/calibration manifest
that independently verified it:

```python
from bayesian_phystwin.prob4d_observation_timestamp_admission import (
    load_claim_bearing_prob4d_observation_timestamp_binding,
)

binding = load_claim_bearing_prob4d_observation_timestamp_binding(
    observation_belief,
    timestamp_lineage_path="outputs/case-a/timestamp-lineage.json",
    expected_timestamp_source_sha256=raw_timestamp_source_sha256,
    timestamp_source_verification_artifact_id=source_manifest_artifact_id,
    bundle_manifest_path="outputs/case-a/factors.json",
    expected_bundle_manifest_sha256=bundle_manifest_sha256,
    row_factor_ids=row_factor_ids,
    metadata={"protocol": "source-frozen-timing-v1"},
)
```

A sidecar content ID alone is insufficient for claim-bearing admission: forged
source bytes and a forged sidecar could remain mutually self-consistent. Merely
passing a digest copied out of that sidecar would not improve the boundary. The
wrapper therefore:

- reads both the timestamp sidecar and factor-bundle manifest through stable
  regular-file descriptors without following symlinks;
- checks device, inode, size, modification time, and change time before and after
  each read, rejecting replacement or in-place mutation;
- limits each snapshot to 64 MiB and hashes the exact bytes;
- writes owner-only private copies and lets the portable parser see only those
  byte-for-byte snapshots, eliminating a hash-then-reopen race;
- requires `source_artifact_sha256` to equal the independently supplied digest;
- requires a separate content-addressed verification artifact and refuses to let
  the timestamp sidecar verify itself;
- binds the source digest, verification artifact ID, sidecar digest, sidecar
  artifact ID, and exact bundle digest into the content-addressed binding; and
- re-snapshots both original files after binding, rejecting any change.

The claim-bearing evidence keys are reserved and cannot be supplied or replaced
through caller metadata. Symlinked and non-regular timestamp or bundle files are
rejected.

`row_factor_ids` is explicit because a factor can contribute several selected
3-D rows, while the timestamp sidecar contains one timestamp record per factor.
The loader requires one factor ID for every BayesianPhysTwin observation row and
checks that each row frame equals its source factor frame. For a claim-bearing
Prob4D path, these IDs should come from the independently validated sparse factor
stack rather than being reconstructed from frame numbers.

The bundle manifest is bound by its exact SHA-256. The consumer rejects schema
drift, duplicate JSON keys, changed source identity, reordered factors, changed
frame indices, unknown row factors, causal-boundary violations, and a
row-to-factor frame mismatch.

## Conditional jitter factor

Given an `(N, 3)` derivative array in metres per second,

```python
jitter_factor = binding.conditional_jitter_low_rank_factor(
    observation_derivative_xyz_per_s
)
```

returns an immutable `(N, 3, F)` factor, with one column per recorded source
factor. Its covariance contribution is `U U^T`. This preserves common timing
jitter among rows originating from the same factor without claiming dependence
between distinct factors.

This factor is an observation-side uncertainty representation. It must not be
added again as diagonal point covariance.

## Shared clock design and prior

```python
from bayesian_phystwin.causal4d_observation_clock_prior import (
    load_causal4d_observation_timing_prior,
)

clock_design = binding.shared_clock_design(
    observation_derivative_xyz_per_s
)
prior = load_causal4d_observation_timing_prior(
    source_only_prior_path,
    expected_artifact_id=(
        binding.shared_clock_offset_prior_artifact_id
    ),
    expected_clock_domain=binding.clock_domain,
    expected_time_scale=binding.time_scale,
)
```

The design has shape `(3N, 1)` and uses the same coordinate flattening as the
physical and nuisance Jacobians. Claim-bearing consumption requires the complete
content-addressed Causal4D prior record, not only its compact Gaussian payload.
BayesianPhysTwin independently checks the closed schema, source-only information
boundary, source execution count and ordering, source offsets, predictive-width
formula and floor, exact content ID, clock domain, time scale, and correction
convention:

```text
aligned_observation_time_s = observation_time_s + offset_s
```

A compact payload containing an artifact ID, mean, and standard deviation cannot
tie those numeric values to that ID. The
`binding.exploratory_shared_clock_prior_from_payload(...)` helper is explicitly
non-claim-bearing. The full-record validator rejects compact payloads and returns
an `ObservationTimingPrior` only after the complete record has been reconstructed
successfully.

Timing identifiability must still be assessed against physical-state, gauge,
visual-bias, and material-lag modes. A source timestamp sidecar and a valid
source-only prior do not by themselves distinguish hardware clock error from
physical relaxation.

## Information-order boundary

Timestamp extraction, clock-domain definitions, conditional-jitter estimation,
the raw timestamp-source digest, its separately frozen verification-artifact ID, the
factor-bundle identity, and a shared-offset prior must be frozen from source or
calibration evidence before confirmation access. The consumer binds those
choices but does not infer or retune them from target outcomes.

## Claim boundary

A valid binding establishes stable byte identity, causal ordering, exact
factor-to-row mapping, separately bound source-byte evidence, and
non-duplicated timing uncertainty semantics. It does not establish timestamp
accuracy, transfer of a timing prior, physical-state identifiability, provider
competence, calibrated target coverage, downstream physical-query improvement,
Causal4D benefit, deployment safety, or state of the art.
