# Deform360 v6 SAM2 checkpoint identity repair

Date: **2026-08-11**  
Status: **pre-source-prediction technical repair; target and confirmation data remain closed**

## What failed

Protected-main workflow run `31456530482` reached the frozen source inventory
and then stopped at `locate-frozen-sam2-checkpoint`. Its compact receipt records:

- status `invalid`;
- error `SAM2 checkpoint identity changed`;
- `10` prepared source objects and `324` prepared source streams;
- `0/10` physical prediction manifests; and
- `0/100` source prediction seals.

The workflow had not opened the development suffix, any v5 confirmation
payload or outcome, or any v6 target payload or outcome. The failure is
therefore a technical runtime-identity failure before prediction, not evidence
against the method.

## Root cause

The reviewed execution contract consistently named and downloaded the original
**SAM 2 Hiera Large** checkpoint:

- model: `sam2_hiera_large`;
- filename: `sam2_hiera_large.pt`;
- official download:
  `https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt`;
- frozen SAM2 source revision:
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`.

However, its frozen checkpoint digest was
`6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`, which identifies
`sam2.1_hiera_small.pt`, not the named original Hiera-Large checkpoint.

The original Hiera-Large checkpoint has:

- SHA-256 `7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b`; and
- byte count `897,952,466`.

Thus the download URL and intended model were unchanged; only the recorded byte
identity was inconsistent.

## Repair design

The original execution amendment remains immutable and stays bound to the
failed run. A separate content-addressed repair is added at:

`protocols/amendments/deform360_official_hub_fresh_object_session_v6_sam2_checkpoint_identity_repair.json`

Its repair ID is:

`28cee70eaa0e8561a320f87d4e51d6c2aad365927814dc94864e299fc145be99`

The repair is limited to
`runtime_sources.sam2_checkpoint_sha256`. It changes no model family or size,
repository revision, source object, camera panel, candidate roster, loss, gate,
fallback, replacement policy, or claim boundary.

The original reviewed runner is retained byte-for-byte at:

`scripts/ci/archive/run_deform360_v6_source_prediction_evidence_v1.sh`

Git blob identity:

`9680176e74e933485e1812bf79b626250925ed1a`

The public runner path now verifies the repair, verifies that archived blob,
patches only the single mismatched SHA in a temporary copy outside the
repository, executes the reviewed runner, and binds the repair ID and corrected
checkpoint identity into the generated execution receipt.

## Failed-run provenance

| Item | Value |
| --- | --- |
| Workflow run | `31456530482`, attempt `1` |
| Source revision | `6111dd42e04b7a050cf7e9e903065d03c79aa2f2` |
| Artifact | `deform360-v6-source-prediction-evidence-31456530482-1` |
| Artifact ID | `9088273849` |
| Artifact digest | `sha256:6acf80b8f1c33d29014ba799eb9331e3ce75bfcc77b6c338e7d9b78eb08e37e9` |
| Execution receipt ID | `3159b09724a0e9082bbf0020c38f0c5ec25c8ce3cc08d92f1eb9fa3418c9316d` |
| Prepared source inventory SHA-256 | `461698eb851413ab5a8a1e702d21432bfbee02ede92c5f9980fb3b8bc0aeebd4` |
| Terminal stage | `locate-frozen-sam2-checkpoint` |
| Physical manifests | `0/10` |
| Source prediction seals | `0/100` |

## Information boundary

The correction was frozen before any physical prediction manifest or source
prediction seal existed. The following remain false:

- development suffix opened;
- future object observations used for prediction;
- v5 confirmation payloads or outcomes opened;
- v6 fresh target selected;
- v6 target payloads or outcomes opened; and
- replacement allowed.

The repaired execution is still authorized only after reviewed merge to
protected `main` on `workstation2`. It must seal the complete 100-record source
prediction batch before any suffix or target stage can be considered.

## Interpretation

This repair does not rescue or alter a scientific result because no scientific
source-prediction result existed. It corrects a pre-outcome runtime identity
mismatch while retaining the original failed receipt. A subsequent run may be
interpreted scientifically only if it reaches the registered source gates with
complete prediction evidence; otherwise its next fail-closed state must again be
classified as technical readiness evidence.
