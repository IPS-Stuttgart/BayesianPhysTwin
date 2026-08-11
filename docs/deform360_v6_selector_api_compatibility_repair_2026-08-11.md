# Deform360 v6 selector API compatibility repair

Date: **2026-08-11**
Status: **pre-source-prediction technical repair; all suffix, confirmation, and target data remain closed**

## Retained failure

Protected-main source run `31522573008` validated the repaired selector byte
identity and reached the first prefix-staging call. The frozen consumer then
called `select_initial_mask_from_rgb`, which the exact pinned
`DeformableObjectSam2VideoPredictor` does not expose. The producer instead
exposes `select_initial_mask`, whose implementation reads one RGB frame and
then performs the same automatic candidate generation, filtering, scoring, and
selection required by the consumer.

The bounded evidence is:

| Item | Value |
| --- | --- |
| Source revision | `e7c303bbac8af462c1437dfc9fb57deaa5537d8f` |
| Workflow run | `31522573008`, attempt `1` |
| Artifact ID | `9113689469` |
| Artifact digest | `sha256:e082d5bccf2d8ac6476396aad9ca0eac9d906521a3069e4867d8457d5ebef417` |
| Receipt ID | `d4012d47004669d4a220e9f57fbac19a3b514da97ca91280ff64a4d9b8922acf` |
| Physical manifests | `0/10` |
| Source prediction seals | `0/100` |

Every information-boundary flag remained false. The failure is technical
readiness evidence, not model-performance evidence.

## Repair

The checksum-bound selector and consumer files remain byte-identical. A
content-addressed process-local adapter supplies only the missing consumer
method. For each already-read exact selected-prefix RGB frame, it temporarily
overrides the selector's frame reader, invokes the existing
`select_initial_mask` method with the declared camera and video metadata, and
restores the original reader in a `finally` block.

This delegates candidate generation, eligibility, scoring, tie-breaking, mask
selection, and diagnostics to the pinned selector implementation. The adapter
fails closed if the producer class, required method absence, delegated method,
frame-reader surface, selector bytes, consumer bytes, or repository revision
changes.

The amendment is:

`protocols/amendments/deform360_official_hub_fresh_object_session_v6_selector_api_compatibility.json`

Repair ID:

`5502830e01585cb1bb208d2d49e05d1f5e1d164dd707c5ff291038949dd0917c`

## Frozen scope

The repair changes no selector or consumer bytes, RGB frame, candidate roster,
camera panel, model, checkpoint, mask-selection algorithm, loss, gate,
fallback, source object, or target policy. It authorizes only a new
protected-main source attempt after review. The required ten physical manifests
and 100 immutable source prediction seals must still exist before any
development suffix can be opened.
