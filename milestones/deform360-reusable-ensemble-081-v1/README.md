# Deform360 reusable-twin ensemble 081 v1

This milestone freezes two post-calibration diagnostics for the first reusable
PhysTwin rope experiment. Both diagnostics are negative for method promotion.

## Source Gibbs gate

- Candidate tuples: 18 finite parent-grid candidates.
- Source episodes: 1, 4, and 6.
- Registered fit frames: `[1, 60)`.
- Selected Gibbs temperature: `0.01`.
- Effective candidate count: `2.376`.
- Gibbs leave-one-action-out relative score: `0.82796`.
- Point-MAP leave-one-action-out relative score: `0.81642`.
- Source gate: **failed**, because the mixture did not match point MAP.

No Gibbs mixture was evaluated on an independent split.

## Exploratory point-MAP result

The source-only trusted point MAP was `50000 / 1 / 50`. It was evaluated only
on the already-open calibration episodes 0, 2, and 8.

| Method | Track RMSE | Symmetric CD |
| --- | ---: | ---: |
| Persistence | 13.695 mm | 13.630 mm |
| Frozen parent | **10.665 mm** | 11.295 mm |
| Trusted MAP, commanded count | 11.037 mm | **11.189 mm** |
| Trusted MAP, supported count | 11.552 mm | 11.330 mm |

The supported-count point MAP is 8.31% worse in track and 0.31% worse in CD
than the frozen parent. The commanded-count version is 3.48% worse in track,
despite a 0.94% CD improvement. Neither is promoted.

## Claim boundary

- Calibration outcomes were already known when this follow-up was designed.
- Calibration evaluation is exploratory mechanism-development evidence only.
- Episode 5 remains sealed and was not read.
- No multi-object, reusable-twin, Bayesian-calibration, or state-of-the-art
  claim follows from this milestone.
- The next evaluation must use fresh objects under a preregistered protocol.

Canonical result hashes:

- source Gibbs posterior: `3ae8878a581abecf90be7a256013d0c2a27c6a47198f1a6a7b94f85c5ccb5186`;
- source trusted point MAP: `8c78103add47ee5e878b7a72d446c9c825102dbdb63b507011d70ce9a9252b55`;
- exploratory evaluation: `5a15a97e1776839f6999b10cf7690238fccd526b87a68dd6e7b284166b5f208d`.
