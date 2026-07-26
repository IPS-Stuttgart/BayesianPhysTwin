# Prior-aware sparse-identity state update: source smoke v1

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

## Evidence boundary

Prediction and scoring are separate commands. Prediction:

- receives the given future controller trajectory;
- overwrites future object observations and visibility before simulator setup;
- masks every future manual identity;
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
`configs/sota/phystwin_prior_aware_sparse_identity_source_v1.json`.

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/florianpfaff/.venvs/deform360-processing-v1/bin/python \
  scripts/remote/run_phystwin_sparse_state_update_source.py predict \
  --protocol configs/sota/phystwin_prior_aware_sparse_identity_source_v1.json \
  --output /home/florianpfaff/bpt-prior-aware-sparse-identity-v1/run/single_lift_cloth
```

After the prediction seal:

```bash
/home/florianpfaff/.venvs/deform360-processing-v1/bin/python \
  scripts/remote/run_phystwin_sparse_state_update_source.py score \
  --protocol configs/sota/phystwin_prior_aware_sparse_identity_source_v1.json \
  --output /home/florianpfaff/bpt-prior-aware-sparse-identity-v1/run/single_lift_cloth
```

No held-v8 artifact or GPU is part of this protocol.
