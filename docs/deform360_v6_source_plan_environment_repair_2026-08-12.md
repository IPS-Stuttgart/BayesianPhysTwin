# Deform360 v6 source-plan environment repair — 2026-08-12

## Retained failure

Protected dual-runtime workflow run `31581551099` completed all ten source
materializations, frame-zero reconstructions, and physical-prior stages. It
retained `10/10` physical manifests and opened no development suffix,
confirmation payload, or target outcome.

The execution then stopped at `materialize-source-plan` because a Python
stdin process read `RUN_ROOT` from `os.environ`, while the predecessor launcher
had only assigned the same deterministic value as a non-exported shell
variable. The bounded failure receipt was
`79bd32e1af16b3529aeb190494c892cfdb927d526a0f1ef0202aafc99c9188cb`.
During legacy receipt enrichment, the dual-runtime lane also exposed that the
older generic receipt layer required `CUDA_HOST_COMPILER_PROBE_PASSED` even
though that legacy probe is not part of the dual-runtime bootstrap.

This is source-independent workflow evidence. The run produced no sealed source
prediction and authorizes no replacement, suffix access, confirmation access,
or claim.

## Repair

The launcher at
`scripts/ci/run_deform360_v6_source_prediction_evidence.sh` is now a
content-addressed wrapper around exact predecessor revision
`812da43f993b4fc5e1f6a96bcc308756b131fc4c` and Git blob
`b2b2307a2f89f3983cce349e1220033bf7f8f50c`.

Before invoking those unchanged predecessor bytes, it:

1. derives `RUN_ROOT` only from the already-bound `RESULTS_ROOT`,
   `AMENDMENT_ID`, and exact `BPT_SOURCE_SHA`;
2. rejects a pre-existing `RUN_ROOT` unless it is byte-for-byte equal to that
   deterministic path;
3. exports the value before any stdin-dispatched Python process is opened;
4. supplies non-observed sentinels only for absent legacy compiler/build-tool
   receipt variables so the predecessor can terminate cleanly;
5. removes the corresponding legacy runtime fields before the final receipt is
   content-addressed; and
6. appends `runtime_source_plan_environment_repair`, binding the predecessor
   source, repair artifact, deterministic relative run root, and removed
   compatibility defaults.

The source-plan implementation, object roster, observations, physical prior,
means, covariance, fallback algorithm, horizons, selector, and target policy are
unchanged.

## Content address

- repair ID: `65096d1d4e8903eeacef0fc50816e47752a61e0d1bb4b6601f291bfcffb9ac4e`
- amendment file SHA-256:
  `1eda9a28e17e46756f9f4bf4fc341b920a8c6f6de8d3e492be1b035a6368651d`
- failed workflow artifact: `9135953420`
- failed workflow artifact digest:
  `sha256:147342a12d05d93378eb652520974a5a001c6e105d083ae6fe707778ca1d165a`

## Scientific boundary

This repair permits only a rerun of the already-authorized prefix-only source
execution. A successful software run would still require 100 sealed source
predictions and the registered source authorization decision before any
development suffix or confirmation payload may be opened. It establishes no
provider competence, covariance calibration, independent-object transfer,
physical-query benefit, deployment safety, or state of the art.
