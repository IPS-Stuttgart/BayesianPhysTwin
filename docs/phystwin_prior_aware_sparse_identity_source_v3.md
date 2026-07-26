# Prior-aware sparse-identity state update: source smoke v3

## Question

The automatic CoTracker3 source arm reached `10.627/20.415` mm and failed its
advancement gate. The stronger `7.891873/13.429357` mm result uses the same
manual identity family in the prefix and in future scoring, so it is an
online-supervised capacity ceiling rather than a deployable comparison.

This smoke asks a narrower, causally clean question:

> Can four material identities observed only during a seven-frame response
> prefix identify a physically propagated state update that improves five
> disjoint hidden material identities in the future?

`single_lift_cloth` is already outcome-open and is used only for development.
Earlier mode diagnostics make it the best available competence case because
low-frequency state updates remain influential in this interaction.

## Why v3 exists

The registered v1 prediction stopped at its replay-parity gate before response
fitting, prediction sealing, outcome scoring, or using future hidden identity
values. The manual-track file had already been loaded and deterministically
split, but no future value entered a diagnostic or decision. V1 is archived as
a technical failure and must not be rerun or scored.

The failure was a missing replay dependency. The state-injection helper copied
initial and restarted Warp states through the training path rather than the
official `pure_inference=True` path, and v1 did not bind that helper's hash.
A target-free diagnosis then established:

- the strength-zero overlay checkpoint is numerically identical to the
  original checkpoint for every simulator parameter;
- historical replay commit `1754a5064869a95e9a1fd1939de10cde5138a13d`
  reproduces the selected raw trajectory exactly;
- after restoring pure-inference state copies, the helper reproduces all 173
  selected-raw frames with zero vector RMSE and zero maximum error and exports
  finite position and velocity states.

V2 corrected that dependency but stopped before output creation or array
loading. Its validator incorrectly required the new selected MatPhys replay to
equal the older released trajectory. The selected replay is exactly
self-consistent, while its intentional difference from the release is
`37.361` mm vector RMSE. V2 only checksum-read its inputs; it initialized no
simulator, fitted no state, and used or scored no future value.

V3 changes only that provenance check: exact parity is required against the
selected trajectory that defines the baseline, while the older-release gap
remains recorded in the hash-bound historical summary. V3 preserves the
identity split, graph basis, response frames, perturbations, priors, gates, and
metrics from v1 and v2. Both predecessors are archived and must not be rerun or
scored.

## Method

Frame-zero farthest-point sampling fixes observed identities `[0, 8, 1, 7]`.
Identities `[2, 3, 4, 5, 6]` are hidden from inference and used only by the
post-seal scorer. Frame-zero nearest-node association is fixed before the
response prefix and maps the four sensors to distinct nodes within `3.55` mm.

The selected raw MatPhys trajectory is replayed with official Warp. A rank-four
symmetric-normalized spring-graph basis includes the global null mode. At frame
114, one-sided finite differences evaluate 24 declared perturbations:

- four graph modes times three position coordinates at a 5 mm step;
- four graph modes times three velocity coordinates at a 0.05 m/s step.

The first four response frames fit a robust Student-t posterior over state
weights and a separate persistent graph-bias field. The remaining three prefix
frames are untouched validation. State-plus-bias must beat an equally bounded
persistent readout field by at least 5% and 0.25 mm.

An accepted update is rerun nonlinearly in Warp. Its prefix displacement must
agree with the finite-difference prediction within both 2 mm vector RMSE and
30% relative RMSE. Position and persistent bias are each capped at 10 mm;
velocity is capped at 0.1 m/s. Failure of the prefix gate, identifiability gate,
replay-parity gate, or nonlinear-closure gate returns the persistence
comparator byte-for-byte.

V3 requires bit-exact selected-baseline replay parity and persists the complete
position/velocity carrier before state inference. The historical replay
summary, strength-zero checkpoint, runtime, state-injection helper, and
implementation files are all checksummed.

## Evidence boundary

Prediction and scoring are separate commands. Prediction:

- receives the given future controller trajectory;
- overwrites future object observations and visibility before simulator setup;
- masks every future manual identity;
- writes replay parity and the replayed position/velocity carrier before
  inference;
- writes a checksummed prediction and correction artifact;
- contains no future metric.

Only after `PREDICTION_COMPLETE` exists may the scorer load future object
observations and the five hidden identities. It reports the unchanged physical
baseline, sparse persistence, state-only rerun, and state-plus-bias candidate.

This is not open-loop SOTA evidence. Four manual prefix identities are an
additional online sensor. A positive result only authorizes a fixed small
source panel; an independent evaluation would require a prospectively locked
sensor and fresh objects.

## Locked execution

The exact inputs, simulator, runtime, source QA, thresholds, and hashes are in
`configs/sota/phystwin_prior_aware_sparse_identity_source_v3.json`.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/florianpfaff/bpt-prior-aware-sparse-identity-v1/runtime-warp111:src \
  /home/florianpfaff/codex-runs/molmomotion-field-davis-timed-20260714/venv/bin/python \
  scripts/remote/run_phystwin_sparse_state_update_source.py predict \
  --protocol configs/sota/phystwin_prior_aware_sparse_identity_source_v3.json \
  --output /home/florianpfaff/bpt-prior-aware-sparse-identity-v3/run/single_lift_cloth
```

After the prediction seal:

```bash
PYTHONPATH=/home/florianpfaff/bpt-prior-aware-sparse-identity-v1/runtime-warp111:src \
  /home/florianpfaff/codex-runs/molmomotion-field-davis-timed-20260714/venv/bin/python \
  scripts/remote/run_phystwin_sparse_state_update_source.py score \
  --protocol configs/sota/phystwin_prior_aware_sparse_identity_source_v3.json \
  --output /home/florianpfaff/bpt-prior-aware-sparse-identity-v3/run/single_lift_cloth
```

No held-v8 artifact or GPU is part of this protocol.
