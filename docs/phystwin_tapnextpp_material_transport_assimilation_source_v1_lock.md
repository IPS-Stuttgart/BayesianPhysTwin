# TAPNext++ Material Transport Assimilation Source Lock

Date: 2026-08-06

Status: 14 prediction inputs staged; all future source outcomes are separately
withheld; no assimilation prediction or future score exists.

## Frozen Implementation

- implementation and protocol commit: `2439c05c`
- protocol file SHA-256:
  `ed6467b2dfe4eb8373b5ebe3fa49e32495f5704b26f0681eb22bd83319ffedbb`
- provider summary file SHA-256:
  `39ac705f49e60219e2fc6e78e31d60db56ce9bb8c719fd4d2745a8ebafd8ca96`
- provider summary canonical result:
  `df5babfc29266a5da9952b5ddb2fb3b148dcf9b33da5c55c1dbde01639bf2637`
- provider source-manifest file SHA-256:
  `e36f57fae7c8c05eb67796088bc28839f420a9eca58c3fa1f940c0d88c71d84f`
- provider source-manifest canonical result:
  `a3443cc082891c978a740a41981e0b3febd1982043548cff9e27868155b28e8f`

## Staged Source Manifest

- artifact kind: `PhysTwinTAPNextPPSparseAssimilationSourceManifest`
- fixed cases: 14
- prediction-input artifacts: 14
- separately withheld future artifacts: 14
- canonical result SHA-256:
  `7a068cb14b99dcd82b0a90b519b29a26c89247d4399df59746f01105910588a6`
- file SHA-256:
  `6a96bc5dbbd1ebfde6d04453cb5c66e918eec0020131fb12fd3d6901b053799c`
- server root:
  `/home/florianpfaff/source-only/bpt-material-transport-assimilation-source-v1`

Every prediction input binds the sealed provider archive, provider report and
seal, provider source result, physical trajectory, graph-construction inputs,
four provider identities, immutable frame-zero material-node indices, and
frame-zero attachment distances. The largest attachment distance is 5.300 mm,
below the frozen 30 mm admissibility limit.

## Sealed Predictions

All 14 prediction runners completed and sealed before any withheld source
future was opened.

- prediction-manifest canonical SHA-256:
  `124da54a2eeb54b772ca730ebfc9be9d49d19e042f8678c1c9a7397b7db62dbb`
- prediction-manifest file SHA-256:
  `3bc387d97961ff5a59b0e4f3326a0fdf60f658cae7b47ade41f0fcffd5ff1075`
- sealed prediction archives: 14
- accepted fixed-material updates: 10
- exact dense fallbacks: 4
- future real outcomes read while predicting: 0

The four fallbacks were triggered because at least one query-frame carrier
lay beyond the frozen 30 mm distance from its immutable frame-zero material
node. No case, attachment, or threshold was changed after this disposition.

## Information Boundary

- Prediction inputs contain only released prefix observations, graph data,
  sealed provider outputs, fixed material attachments, and the physical
  rollout.
- Future point clouds and future manual tracks exist only in separately hashed
  withheld artifacts.
- All 14 cases remain mandatory; replacement and future-based tuning are
  forbidden.
- No assimilation prediction, future source metric, independent target, or
  held-v8 artifact was opened when this lock was written.
