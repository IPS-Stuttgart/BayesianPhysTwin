# MatPhys part-aware reconstruction control

## Scope

This is a deliberately noncausal capacity test on the already-open
`single_lift_sloth` case. It fits all released frames, including the future
evaluation interval. It cannot support a predictive or state-of-the-art claim.

The method is not the published MatPhys method. MatPhys does not release its
final per-case semantic bundle, so Bayesian-PhysTwin supplies a deterministic
DINO graph partition and a new part-conditioning adapter. The adapter projects
each part descriptor into the released decoder's material embedding. Both the
proxy bytes and adapter implementation are bound by the training audit.

## Pre-execution correction

The earlier `matphys-official-reconstruction-control-v1` protocol loaded the
graph-part proxy but failed to install the adapter. Because all parts of this
case share one material row, that runner would have ignored the DINO
descriptors and duplicated the completed one-part public-artifact control. No
training or outcome was produced under that protocol.

The replacement protocol is
`configs/sota/matphys_part_aware_reconstruction_control_v1.json`. It requires
the `simple-videomae-dino-part-conditioning-v1` adapter at positive scale in
both training and export. Validation rejects a checkpoint whose audit does not
bind that contract.

A second source-independent preflight found that the staged byte-bound proxy
uses the registered compact edge-semantics contract. Version 1.1 names that
contract explicitly. Its `train_ready.pt`, 1024-dimensional part descriptors,
assignments, material distributions, and graph provenance are inherited from
the full proxy; only node/edge semantic tensors unused by the simple decoder
are compacted. No training or outcome was produced under version 1.

The v1.1 smoke then stopped during import, before model construction, because
Warp 1.15 no longer provides the private `warp._src.utils.warn` helper expected
by pinned MatPhys. Version 1.2 installs a signature-compatible warning adapter
before importing MatPhys and records the failed launch log hash in the protocol.
That launch produced no checkpoint, metric, or scientific outcome.

The v1.2 retry passed import but stopped before model construction because the
staged proxy summary selected one case while its material mapping still listed
four. Version 1.3 requires both the proxy records and mapping to contain exactly
the requested case. The v1.2 launch likewise produced no checkpoint, metric, or
scientific outcome.

The v1.3 retry reached the upstream dataset splitter, which computes
`floor(0.8 * 1) = 0` and tries to construct a shuffled empty loader before its
existing `--case_name` branch replaces both loaders with the selected case.
Version 1.4 binds a narrow compatibility wrapper that sets only this provisional
split to one case. The upstream case-specific loaders and all training targets
remain unchanged. The v1.3 launch produced no checkpoint, metric, or scientific
outcome (log SHA-256
`ca8d2a5f13883350af10288b91b8f97a8bb63fac05b648bab8b0b049ac2cd92b`).

The v1.4 retry constructed the part-aware model, then stopped before its first
objective or optimizer step because the runner inherited the causal
source-panel boundary guard. Version 1.5 replaces that mismatched hook with an
exact full-sequence reconstruction guard: it requires `--fit_all_frames`, binds
the released `frame_len`, and records the same objective end on every call. The
v1.4 launch produced no checkpoint, metric, or scientific outcome (log SHA-256
`ff6f00cb9930bf8bb62aa957a55d7e9efc7feede8d28f6a699443f0928dfee5c`).

## Decision

Run the separately locked
`configs/sota/matphys_part_aware_reconstruction_smoke_v1.json` one-epoch
protocol first to verify finite optimization, full future-access labeling,
strict checkpoint export, and official metric evaluation. Continue to the
fixed 200-epoch terminal checkpoint only if the smoke passes mechanically.

A positive reconstruction result would show capacity only. A later predictive
experiment would need a separately frozen source-only training design, target
prefix selection, pinned-PhysTwin replay parity, and an independent cohort.
