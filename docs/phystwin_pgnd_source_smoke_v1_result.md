# PGND source-backbone competence result

## Status

The frozen single-case source gate failed. Raw PGND replacement is closed
without a wider run.

This result concerns one already-open `single_lift_sloth` source case and the
specific frozen bridge in commit
`32b1a2af5a03a23eea453e1c5316cf2d3e9ff097`. It is not an independent
benchmark, a confirmation result, or a general claim about PGND.

## Evidence boundary

The prediction carrier contained only:

- the unchanged PhysTwin trajectory;
- known controller actions;
- frame-zero surface-layout counts;
- the released train/test split.

Future object coordinates and manual tracks were absent from the carrier.
They were loaded only after two complete PGND replays agreed bit-for-bit and
the prediction was sealed.

The executed dependencies were:

- Bayesian-PhysTwin adapter commit
  `32b1a2af5a03a23eea453e1c5316cf2d3e9ff097`;
- PGND commit `ae050d1342faa0bceb2a10f4b0ab2e11682351cb`;
- public sloth checkpoint SHA-256
  `1ce7f86a40058c2680784ac40f633a67e00e9ce8af8a6111acc3362d71d3b052`;
- PGND configuration SHA-256
  `01e33f8152b25ac80998f0ddaadd182f88ca96b18432bc288edc4723a943a915`.

Both Git checkouts were clean. Two complete GPU replays had a maximum absolute
position difference of exactly `0.0 m`.

## Aggregate result

All values are millimetres. CD is the released one-way visible-observation to
predicted-surface L1 nearest-neighbor metric. Track is the released frame-zero
manual-identity readout followed through the test interval.

| Method | Future CD | Future track |
| --- | ---: | ---: |
| Endpoint persistence | 25.918 | 49.967 |
| Equal-support PhysTwin | 22.529 | 28.482 |
| Full PhysTwin | **18.494** | **29.420** |
| Raw PGND candidate | 24.172 | 49.580 |

Relative to full PhysTwin, PGND regressed by:

- `+30.70%` in future CD;
- `+68.53%` in future manual-track error.

Relative to the equal-support PhysTwin particle subset, PGND regressed by
`+7.29%` CD and `+74.07%` track error. Relative to exact endpoint persistence,
it improved CD by `6.74%` and track error by only `0.77%`.

## Horizon behavior

Relative PGND changes versus full PhysTwin were:

| Horizon | CD | Track |
| --- | ---: | ---: |
| Early | +39.83% | +13.53% |
| Middle | +33.94% | +59.94% |
| Late | +23.74% | +111.48% |

The late identity error is the clearest failure. PGND produces nontrivial
motion, but the generic plush dynamics and one-gripper bridge do not preserve
the source object's material identities as well as its fitted PhysTwin.

## Decision

The preregistered gate required at least a 2% improvement in both aggregate CD
and aggregate manual-track error relative to full PhysTwin. Neither metric
passed. Therefore:

- do not run the wider source panel;
- do not tune scale, orientation, contact selection, blending, or residual
  caps on this opened case;
- retain PGND as a documented executable external-backbone control;
- keep the primary development effort on guarded Bayesian state/discrepancy
  updates around the fitted physical twin.

## Artifacts

The committed compact artifacts are under
`results/sota/phystwin_pgnd_source_smoke_v1/`.

- `prediction_input_summary.json` SHA-256:
  `719a8813475b61065262eb01532045669e815e4fc2b2ab323bfdff500fa1b768`
- `prediction_seal.json` SHA-256:
  `5dd77f354370a489a1d8a73886c5483c8af8c9536f0196b87811a63d9701f75c`
- `evaluation.json` SHA-256:
  `ab7cadd1951943b151e617a2aad80c684ec6977befe39de7440f087524d3c7ab`
