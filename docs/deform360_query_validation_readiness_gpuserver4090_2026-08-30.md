# Deform360 query-validation readiness on `gpuserver4090`

Date: **2026-08-30**

## Outcome

A file-triggered GitHub Actions run completed successfully on the self-hosted
runner label `gpuserver4090` against the read-only Deform360 staging tree at:

```text
/mnt/seagate10tb/florianpfaff/datasets/deform360
```

The retained run is GitHub Actions run `33304040922`, revision
`827e9d1ab42e192e6fcb43e5e60f23c4d632faf9`. The self-hosted job executed on
the registered runner name `workstation1` with two NVIDIA RTX 4090 GPUs. The
uploaded artifact is `deform360-query-readiness-33304040922-1`, artifact ID
`9729872170`, SHA-256
`91f85d95f2d379fb97dde2990110ff1a587ce267c4e68c6a12f8048bf3106d05`.

The decision is:

```text
development-metadata-design-ready
```

This is useful progress toward a public-data validation, but it is not a model
result and does not authorize target scoring.

## What the run did

The workflow first ran hosted tests and formatting checks. The self-hosted stage
then:

1. verified the exact mounted data root;
2. rebuilt the existing target-blind names-only inventory;
3. verified byte identity of the strongest retained historical hash-only
   exclusion union;
4. selected eight objects deterministically from names only;
5. opened only their released `metadata.json` files;
6. selected one source and one target episode with different registered action
   families for each object; and
7. uploaded aggregate readiness evidence before deleting scratch files.

It did **not** decode camera media or open robot arrays, tactile arrays, geometry,
tracks, target futures, or score-bearing outcomes.

## Dataset and exclusion state

The names-only inventory remained stable relative to the earlier
`gpuserver4090` preflight:

- recognized objects: `168`;
- inventory content identity:
  `cb37c193052baae41140d0cb48c642c16e3840a850ca104a4883555eb19f7611`;
- reserved target objects present by name: `12`;
- historical exclusion hashes: `216`;
- objects absent from that historical union: `47` (`41` sheet and `6`
  volumetric).

Absence from the historical union means only **provisionally unexcluded**. The
union is bound through its historical cutoff, but a complete post-cutoff
cross-project exposure delta has not yet been established. Therefore the run
cannot call any object fresh or authorize a fresh-confirmation claim.

## Deterministic development design

| Object | Stratum | Source episode | Target episode | In historical exclusion |
| --- | --- | --- | --- | --- |
| `038-mat-cloth` | sheet | `3`: drag corner | `9`: fold | no |
| `038-black-bag-cloth` | sheet | `8`: drag edges | `5`: lift corners | no |
| `087-plastic-bag-blue-cloth` | sheet | `8`: lift opposite edges | `3`: fold corner | yes |
| `144-jar-opener-cloth` | sheet | `6`: lift sides opposite | `2`: drag side | no |
| `072-cotton-clohesline` | volumetric | `2`: drag edge | `0`: lift edge | yes |
| `102-stress-ball` | volumetric | `8`: squeeze | `6`: lift whole | yes |
| `053-squeezer` | volumetric | `6`: lift middle bottom | `4`: squeeze | yes |
| `063-flower` | volumetric | `9`: wave flower and leaf | `7`: drag | yes |

All eight selected metadata files were readable and contained a valid
source-target pair under the frozen action-family vocabulary. There were no
unsupported selected objects and no replacement was performed.

This roster is for adapter and source-target contract development. The
historically excluded members are natural development cases. The three objects
not found in the historical union must not be promoted to fresh confirmation
until the current exclusion union is complete; future payload-opening protocols
may instead preserve them and use exposed objects for development.

## Next admissible stage

Before any target future can be opened, a separate committed protocol must bind:

- either an exact released processed-annotation adapter or an exact revision of
  the official processing pipeline;
- allowed paths and representation semantics for every selected source and
  target episode;
- the physical hypothesis bank and registered query signatures;
- quotient construction, prior weights, and Jeffrey-lift implementation;
- source-only candidate, covariance, and guard fitting;
- physical fallback and deterministic residual-persistence comparators;
- object-level proper score, harm rule, and relation-breaking controls;
- a complete current exclusion union if a fresh-confirmation claim is intended;
  and
- a prediction seal before target-future scoring.

The current decision fields remain:

```json
{
  "development_metadata_design_ready": true,
  "fresh_confirmation_authorized": false,
  "target_payload_access_authorized": false,
  "model_scoring_authorized": false
}
```

## Claim boundary

This execution establishes only that a stable public Deform360 staging tree can
support a deterministic metadata-level source-target design on
`gpuserver4090`. It is not a payload-integrity certificate, model evaluation,
real-data accuracy result, calibration result, physical-transport result, or
proof that the registered quotient is correct.

The machine-readable retained record is
`evidence/deform360/query_validation_readiness_gpuserver4090_2026-08-30.json`.
