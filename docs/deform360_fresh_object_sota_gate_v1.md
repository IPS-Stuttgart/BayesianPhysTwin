# Deform360 fresh-object SOTA gate v1

## Status

The source-only phase is complete. Fourteen of 18 queued objects passed, all
four technical failures were preserved without retry or replacement, and the
deterministic 12-object cohort is locked at
`results/sota/deform360_fresh_source_lock_v1/deform360_fresh_object_cohort_lock_v1.json`
with internal SHA-256
`bafe26848ee83d8a4201e9d11d51af106370647f76ec702003e9ec51d3843729`.
Predictions and outcomes have not been opened by this protocol.

## Objective

Test whether the unchanged pairwise-consensus Bayesian state update transfers
to genuinely fresh Deform360 objects and, if an authoritative evaluator
contract becomes available, whether it improves the official per-episode 3D
state of the art.

The open-development evidence justifying this experiment is the passing
candidate-metric audit at:

```text
results/sota/deform360_candidate_metric_sensitivity_v1/source_audit.json
```

The audit is not confirmation and is not an official comparison.

## Frozen method

The candidate is:

```text
raw_selected_backbone_full_blend_rbf_pairwise_clique
```

It binds to implementation commit
`e2f8d827bfd60df79eeffee511a5df7e2d53ea21` and source protocol
`deform360-open27-raw-alltracker-pairwise-gate-v1-development`.

The fresh run must preserve:

- the selected physical/persistence backbone rule;
- the pairwise-consensus thresholds and RBF configuration;
- the same update timing and causal RGB-prefix boundary;
- bit-exact fallback when the correspondence gate rejects;
- no outcome-dependent method, threshold, rank, or camera selection.

Any method change requires a new version and a different fresh cohort.

## Prerequisites

Before cohort selection:

1. Run the source validator on metadata and frame-zero inputs only.
2. Reject malformed enums, missing streams, insufficient episode length,
   insufficient camera support, or backend-inadmissible geometry.
3. Bind one point-count contract across protocol and physical backend.
4. Obtain a hash-only exclusion manifest from the owners of all prior or
   reserved evaluations; do not inspect their targets.
5. Select physical objects, not individual favourable episodes.
6. Publish the ordered object/episode manifest and all source hashes.
7. Bind the exact code commit, environment, model checkpoints, and evaluator
   contract before any future outcome is read.

The source admission additionally requires the released `split.json`
`frame_len` to equal `control_points.meta.json` `num_active_frames`. The public
generator sizes the split from the complete contact window while dropping
inactive frames from `final_data`; episodes where those counts differ are
inadmissible because the split can index beyond the actual trajectory.

Technical failures remain in the denominator and are reported separately as
successful predictions, retained technical failures, and unsealable cases.
Cases are never silently replaced.

## Cohort

Use at least 12 fresh physical objects, stratified across rope, cloth, soft
toy, and other backend-admissible deformables where source metadata permits.
Use a fixed metadata-only episode rule per object. More episodes may increase
precision, but the unit of generalization and bootstrap resampling is the
physical object.

The final cohort size and ordered manifest must be frozen by a separate lock
artifact. This draft intentionally contains no target-derived membership.

## Arms

Run and retain:

1. sealed physical prior;
2. exact persistence;
3. selected physical/persistence backbone before the online update;
4. frozen pairwise-consensus update;
5. exact-fallback indicator and all gate diagnostics.

When runnable under the same authoritative contract, also evaluate released
ParticleFormer and PGND checkpoints. Published table values alone are context,
not locally reproduced baselines.

## Metrics

### Official primary metrics

Use future Chamfer distance and future track error exactly as specified by a
released Deform360 evaluator or content-hashed author contract. The contract
must resolve every field in `Deform360Official3DParityContract`.

No official SOTA claim is allowed if that contract remains unresolved.

### Fixed secondary metrics

Regardless of official-evaluator availability, report the explicit
hidden-identity population used by the source audit:

- mean Euclidean identity error;
- one-sided prediction-to-target Euclidean Chamfer;
- symmetric Euclidean Chamfer;
- coordinate MSE and RMSE;
- squared and Euclidean Chamfer variants;
- frame-, episode-, and object-balanced aggregation.

All units must be explicit.

## Gates

The method passes fresh transfer only when:

1. both object-balanced official primary metrics improve over the strongest
   locally reproduced baseline;
2. physical-object cluster 95% intervals for both paired differences exclude
   zero in the favourable direction;
3. at least 8 of 12 physical-object means improve on both primary metrics;
4. no object has more than a 20% regression on either primary metric;
5. the three explicit metre-valued secondary metrics improve over both the
   physical prior and persistence under every aggregation convention;
6. all locked cases are accounted for and no replacement occurred;
7. the exact-fallback and causal information-boundary validators pass.

An official SOTA statement additionally requires:

1. a parity-ready authoritative evaluator contract;
2. local reproduction of the strongest eligible baseline under that contract;
3. a better score than that reproduced baseline on both official primary
   metrics.

Comparing a local number directly with a published table entry is not enough.

## Execution order

```text
source-only admission and exclusion manifests
-> ordered cohort lock
-> all candidate and baseline prediction seals
-> completeness barrier
-> one outcome-opening operator
-> official and fixed-secondary scoring
-> immutable result artifact
```

If the authoritative evaluator remains unavailable, the run may proceed under
the fixed secondary contract. Its strongest allowed claim is then:

> Fresh-object transfer under explicit candidate metric conventions.

It must not be described as official Deform360 parity or state of the art.

## Current recommendation

Proceed with source-only cohort preparation in parallel with an evaluator
request to the Deform360 authors. Do not change the method. The open evidence
shows enough metric-robust headroom to justify the cost of a fresh run; the
remaining obstacle to a SOTA claim is evaluation parity and independent
transfer, not another retrospective model variant.
