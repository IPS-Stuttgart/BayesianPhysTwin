# TAPNext++ Sparse Assimilation Source Lock

Date: 2026-08-06

Status: eight prediction inputs staged; future source outcomes withheld; no
assimilation prediction or future score opened.

## Frozen Implementation

- implementation commit: `4c495f66`
- protocol file SHA-256:
  `303654f662fb0852b2c02dcfe5d7235992a252c3bc3b8798c8e05c80e216b0c0`
- provider transfer summary file SHA-256:
  `ed555519be8f5bcc6e8a3734b9ce536f3692f4a1ee7a9bd97392e9861b7ae1d9`
- provider transfer canonical result:
  `08e24ba766b2904312bfc8898fb9bfd92ffeae37e3e9de544a864ceff7fe8dc6`

## Staged Source Manifest

- artifact kind: `PhysTwinTAPNextPPSparseAssimilationSourceManifest`
- fixed cases: 8
- prediction-input artifacts: 8
- separately withheld future artifacts: 8
- canonical result SHA-256:
  `881e725d7728bf54e4323240f7bf1827415a36e82f59820ab16dd333b4114dac`
- file SHA-256:
  `db27105334854a8ffa40d7b370f539f53e1dbe740fcd6216435ef62cc09bccb2`
- server root:
  `/home/florianpfaff/source-only/bpt-tapnextpp-sparse-assimilation-source-v1`

The compact manifest is archived under
`results/sota/phystwin_tapnextpp_sparse_assimilation_source_v1/source_lock/`.

## Pre-outcome Technical Amendments

Two staging attempts stopped before writing any case artifact:

1. `optimal_params.pkl` was initially resolved under the released data
   directory rather than the physical rollout directory. Commit `7986e8ce`
   corrected the source path.
2. the provider source-frame index was initially requested from the completion
   report, while the frozen index is bound by the per-case tracker protocol.
   Commit `4c495f66` now validates and hashes that protocol explicitly.

Neither abort created a prediction input, a withheld artifact, a prediction,
or a metric. No numerical setting, case, arm, or advancement gate changed.

## Information Boundary

- Prediction inputs contain released prefix observations, static graph data,
  sealed provider output, and the full physical model rollout.
- Future real point clouds and manual tracks exist only in the separately
  hashed withheld artifacts.
- Failed provider cases cannot be replaced and must use exact dense fallback.
- No independent target or held-v8 artifact is authorized or accessed.

