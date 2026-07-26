# Penguin CoTracker3 Bias-Aware Source Result v1

Date: 2026-07-26

Status: source gate failed safely; no fresh-object evaluation is authorized.

## Question

This registered same-object source study tested whether causal, exact-prefix
CoTracker3 material identities could support the existing bias-aware Bayesian
state update. The update was allowed to modify the selected reusable physical
response only when sparse multiview motion also agreed with the
action-conditioned physical response.

Predictions for penguin episodes 1, 3, 4, 6, 7, and 9 were produced and
hashed before any source PCD outcome was opened. All six prediction jobs
completed without a technical fallback. The cohort seal records that
`pcd_clean` was not read during prediction.

## Result

No one of the 18 registered prefix windows passed the frozen dynamic-evidence
gate. Consequently, the unguarded candidate and the leave-one-episode-out
guarded arm both remained bit-exact copies of the physical baseline in every
episode.

| Arm | Hidden identity RMSE | Hidden Chamfer | Late identity RMSE | Late Chamfer |
| --- | ---: | ---: | ---: | ---: |
| Selected physical baseline | 8.347 mm | 6.856 mm | 11.501 mm | 9.054 mm |
| Unguarded bias-aware update | 8.347 mm | 6.856 mm | 11.501 mm | 9.054 mm |
| LOO-guarded bias-aware update | 8.347 mm | 6.856 mm | 11.501 mm | 9.054 mm |

The registered changes are therefore exactly `0.00%` in all four aggregate
metrics. Maximum episode-metric degradation is also exactly `0.00%`.

The transfer gate fails because:

- aggregate identity and Chamfer improvements are below 5%;
- late identity and Chamfer improvements are below 5%;
- no held-episode update is accepted.

The non-regression gate passes only because exact fallback was used.

## Why The Update Abstained

The target-free dynamic gate has five necessary conditions. Across the 18
episode-window pairs, the numbers satisfying each condition were:

| Necessary condition | Passing windows |
| --- | ---: |
| At least 9 available identities | 11 / 18 |
| At least 3 motion identities | 16 / 18 |
| Physical response at least 0.5 mm | 15 / 18 |
| Observed motion at least 0.5 mm | 17 / 18 |
| Robust physical-agreement gain at least 0.40 | 1 / 18 |
| **All conditions jointly** | **0 / 18** |

The sole window with sufficient physical agreement was episode 1 at frame
19. Its physical response was only `0.025 mm`, so it correctly failed the
minimum-response gate. All other physical-agreement gains were at most
`0.182`.

Thus the failure is not simply an absence of triangulated points. Sparse
multiview observations were available in most windows, but their motion was
not supported by the selected action-conditioned physical response. Updating
from that evidence would reintroduce the coherent-bias failure observed in
the camera-only prospective study.

## Decision

Do not scale this fixed CoTracker3 feeder to a fresh-object or confirmatory
cohort. Do not relax its physical-agreement threshold against these opened
source outcomes.

The result supports the narrower Bayesian-PhysTwin design principle:

> A camera-derived discrepancy update needs both structurally redundant
> observation evidence and causal support from the action-conditioned
> physical belief; otherwise it must leave the baseline unchanged.

This experiment does not show that CoTracker3 identities are inaccurate in
general. It rejects the fixed combination of exact-prefix CoTracker3
triangulation, the selected candidate-03 physical response, and the frozen
source-v4 dynamic-evidence gate on this same-object source panel. A future
observation feeder needs an independent modality or materially stronger
source-validated identity evidence before a new prospective protocol is
justified.

## Information Boundary

- Only RGB frames through each update endpoint were decoded during
  prediction.
- No full-window tracker artifact or future RGB frame formed a prediction.
- No PCD outcome or source metric was read before the six-case cohort seal.
- The queried center identities were excluded from both hidden metrics.
- No held-v8 artifact, fresh object, or sealed target was accessed.
- This is same-object episode-held-out development evidence, not
  object-held-out transfer, calibration, or state-of-the-art evidence.

## Provenance

- Bayesian-PhysTwin prediction commit:
  `5b7bb15ceabca40841edc796b79bd8b47be4f406`
- CoTracker source commit:
  `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`
- CoTracker checkpoint SHA-256:
  `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`
- prediction cohort-seal file SHA-256:
  `df3f502e907107a6f6aa1d0d1ca2f10beff0a474db9489b1fdb6c9aafd614278`
- prediction cohort-seal canonical result SHA-256:
  `efef2f111f3a5b1af5c8e77702fb6a4bf9e38893ae45387aa59634811f4eba38`
- evaluation file SHA-256:
  `d25a6eccdb0f5d082718a95a768d239aa69f027e26818346df6cfa7b1165b6f7`
- evaluation canonical result SHA-256:
  `f0944f544786293f50d4007000a79802816bb630ed44ad52d489a9e3a2a5e507`
- archived evidence:
  `results/sota/diagnostics/deform360_penguin_cotracker_bias_source_v1/`
