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
   back to its exact source factor and can form one low-rank timing factor per
   recorded factor. Rows from the same factor therefore retain their common
   timestamp perturbation, while different factors are independent under this
   declared representation.
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

## Loading and binding

```python
from bayesian_phystwin.prob4d_observation_timestamps import (
    load_prob4d_observation_timestamp_binding,
)

binding = load_prob4d_observation_timestamp_binding(
    observation_belief,
    timestamp_lineage_path="outputs/case-a/timestamp-lineage.json",
    bundle_manifest_path="outputs/case-a/factors.json",
    expected_bundle_manifest_sha256=bundle_manifest_sha256,
    row_factor_ids=row_factor_ids,
    metadata={"protocol": "source-frozen-timing-v1"},
)
```

`row_factor_ids` is explicit because a factor can contribute several selected
3-D rows, while the timestamp sidecar contains one timestamp record per factor.
The loader requires one factor ID for every BayesianPhysTwin observation row and
checks that each row frame equals its source factor frame.

The bundle manifest is read from an exact SHA-256 snapshot. The consumer rejects
schema drift, duplicate JSON keys, changed source identity, reordered factors,
changed frame indices, unknown row factors, causal-boundary violations, and a
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
clock_design = binding.shared_clock_design(
    observation_derivative_xyz_per_s
)
prior = binding.shared_clock_prior_from_payload(source_only_prior_payload)
```

The design has shape `(3N, 1)` and uses the same coordinate flattening as the
physical and nuisance Jacobians. The prior payload must match the exact artifact
ID, clock-domain name, and correction convention declared by the lineage:

```text
aligned_observation_time_s = observation_time_s + offset_s
```

The resulting `ObservationTimingPrior` can be passed to the explicit timing
nuisance machinery. Timing identifiability must still be assessed against
physical-state, gauge, visual-bias, and material-lag modes. A source timestamp
sidecar does not by itself distinguish hardware clock error from physical
relaxation.

## Information-order boundary

Timestamp extraction, clock-domain definitions, conditional-jitter estimation,
and a shared-offset prior must be frozen from source or calibration evidence
before confirmation access. The consumer binds those choices but does not infer
or retune them from target outcomes.

## Claim boundary

A valid binding establishes byte identity, causal ordering, exact factor-to-row
mapping, and non-duplicated timing uncertainty semantics. It does not establish
timestamp accuracy, transfer of a timing prior, physical-state identifiability,
provider competence, calibrated target coverage, downstream physical-query
improvement, Causal4D benefit, deployment safety, or state of the art.
