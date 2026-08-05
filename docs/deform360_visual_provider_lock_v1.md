# Deform360 visual-provider and calibration locks v1

## Purpose

The official-Hub Deform360 visuotactile protocol selected fresh calibration and
confirmation objects using names and `metadata.json` only. Its visual arms already
require persistent identities and a complete joint gauge prior, but those phrases
do not identify one executable visual producer. Different Prob4D revisions,
MotionCrafter model bytes, stochastic schedules, window geometry, metric-frame
priors, or covariance truncation can materially change the result.

The additive target-blind amendment at
`protocols/amendments/deform360_official_hub_visuotactile_v1_visual_provider_lock.json`
closes that ambiguity without changing the selected cohort or opening selected raw
payloads.

## Two locks and one information order

The protocol uses two separately content-addressed artifacts:

1. `Deform360VisualProviderLockV1` is committed before selected calibration
   payloads are downloaded. It fixes the exact Prob4D and MotionCrafter revisions,
   provider manifest and attestation bytes, immutable model-set identity, seed
   policy, window geometry, dense-storage dtype, initial metric-frame prior,
   additional-anchor policy, and gauge-covariance retention policy.
2. `Deform360VisualCalibrationLockV1` is committed after calibration objects are
   processed but before confirmation payloads are opened. It binds the visual,
   contact-anchor, deployment-guard, and interval-calibration artifacts, together
   with the exact calibration objects and achieved finite-sample rank.

```text
metadata-only Stage 0 selection
        |
        v
commit visual-provider lock
        |
        v
open calibration objects only
        |
        v
commit visual/contact/guard/interval calibration lock
        |
        v
open confirmation objects exactly once
```

Both loaders reject duplicate JSON keys, unknown fields, non-finite JSON,
nonliteral revisions or SHA-256 identifiers, changed content addresses, and any
record claiming that target outcomes or forbidden payloads were already opened.

## Provider semantics

The v1 provider lock deliberately admits one narrow path:

- `IPS-Stuttgart/Prob4D` provider API 2;
- causal stream contract 2;
- persistent material identities;
- the complete joint cross-window gauge covariance;
- exclusive causal cutoffs;
- one mandatory initial metric-frame prior; and
- either no additional metric anchor or a separately labelled independent sparse
  anchor.

The initial metric-frame prior is not described as “no metric anchor.” An arm
with no additional anchors is still metric-initialized. Any independently sparse
anchor is a sensor-assisted arm and must remain labelled separately.

## Python API

```python
from bayesian_phystwin import (
    Deform360VisualProviderLockV1,
    save_deform360_visual_provider_lock,
)

lock = Deform360VisualProviderLockV1(
    provider_revision=prob4d_revision,
    provider_manifest_id=provider_manifest_id,
    provider_attestation_sha256=provider_attestation_sha256,
    motioncrafter_revision=motioncrafter_revision,
    model_set_id=model_set_id,
    root_seed=root_seed,
    seed_policy="per-object-derived-seed-v1",
    window_size=25,
    overlap=8,
    height=320,
    width=640,
    storage_dtype="float32",
    initial_metric_frame_prior_id=initial_metric_frame_prior_id,
    additional_metric_anchor_policy="none",
    max_gauge_rank=64,
    minimum_retained_gauge_trace=0.999,
)
save_deform360_visual_provider_lock("visual-provider-lock.json", lock)
```

After calibration, create `Deform360VisualCalibrationLockV1` with one calibration
ID for each complete visual, contact-anchor, guard, and interval bundle. The
object list is the statistical-unit list: repeated frames, tracks, views, and
taxels do not increase `calibration_group_count`.

## Finite-sample reporting

The selected Stage-1 cohort contains ten calibration objects. A nonrandomized
finite-group conformal rule therefore has increments of `1 / (10 + 1)`. The lock
records the concrete rank used by the experiment rather than allowing prose such
as “90% interval” to hide an unattainable or changed rank. Coverage of the whole
deployed policy, including exact physical fallback, is primary; coverage
conditional on acceptance is a selection-affected diagnostic unless separately
registered.

## Claim boundary

These locks establish information order, executable method identity, and
portable provenance. They are not evidence that Prob4D observations are accurate,
that contact anchors are informative, that the Bayesian update improves a
physical query, that predictive intervals are calibrated, that exact fallback is
a universal safety theorem, or that Causal4D interventions improve.

## Finite-group calibration binding

The Stage-1 calibration lock must bind the exact content identity of
`deform360_official_hub_visuotactile_v1_calibration_separation.json`.
That design fixes the physical object as the statistical unit, pools all
ten calibration objects for the primary nominal-90% interval, records
conformal rank 10, and forbids a nominal-95% or five-object-stratum
nominal-90% split-conformal claim.

The deployed predictor, score, guard, grouping rule, and endpoint set
must be fixed from external or source-only evidence before interval
scores are inspected. Calibration outcomes cannot also select the policy
under this split-conformal contract. CV+ or jackknife+ would require a
separately versioned design.
