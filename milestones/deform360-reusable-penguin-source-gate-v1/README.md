# Deform360 reusable-PhysTwin penguin source gate v1

This milestone records the frozen, source-only reusable-twin selection on
Deform360 object `171-penguin`. It is a negative transfer result and does not
authorize evaluation on held episodes.

## Information boundary

- Source episodes `1,3,4,6,7,9` were used for fitting and source diagnostics.
- Held episodes `0,2,5,8` were not read, predicted, scored, or opened.
- Every source outcome used an already sealed future reveal and validated
  strict-hull reconstruction.
- The source selector used the registered 18-candidate physical grid and the
  frozen trust rule. Candidate physics could not alter the trust decision.
- The alpha oracle was run only after the frozen source gate failed. It is
  nondeployable, did not refit trust or physics, and cannot authorize held
  evaluation.

## Frozen source result

The pooled selector chose candidate 3:

```text
init_spring_y  = 10000
drag_damping   = 3
dashpot_damping = 100
```

Against exact persistence, the selected reusable twin produced:

| Evaluation | Track RMSE change | Chamfer change | Wins / ties |
| --- | ---: | ---: | ---: |
| Pooled source fit | +39.81% worse | +36.71% worse | 0 / 1 |
| Leave-one-action-out | +60.73% worse | +56.17% worse | 0 / 1 |
| Per-episode physical-candidate oracle | +32.17% worse | +24.66% worse | 0 / 1 |

Even selecting the best physical candidate separately for each source episode
does not beat persistence. The registered physical family therefore fails the
source gate, independently of cross-action pooling.

The frozen trust rule accepted response gains of `1.2` on episodes
`1,3,6,7`, `0.2365` on episode `4`, and returned exact fallback on episode
`9`. The accepted gains over-transmit the simulated response.

## Post-gate diagnosis

A source-label alpha oracle over
`{0,0.025,0.05,0.1,0.2,0.4,0.8,1.2}` found:

| Episode | Best candidate | Best alpha | Relative score |
| --- | ---: | ---: | ---: |
| 1 | 15 | 0.100 | 0.9713 |
| 3 | 13 | 0.200 | 0.9363 |
| 4 | 0 | 0.025 | 0.9943 |
| 6 | 5 | 0.200 | 0.9686 |
| 7 | 0 | 0.000 | 1.0000 |
| 9 | 16 | 0.400 | 0.9509 |

This oracle improves mean track RMSE by 5.40% and Chamfer by 2.80%, with five
episode wins and one exact fallback. The optimal candidate and response scale
vary substantially by action, and one action rejects every update. This is
headroom evidence for a guarded, baseline-relative belief update, not evidence
that the current reusable-twin method transfers.

## Decision

The frozen source gate failed. No penguin held episode is authorized.

Do not tune the current selector on these six source outcomes or open the four
held episodes. The next reusable-twin attempt, if pursued, needs a newly locked
method with source-calibrated regret control and exact persistence fallback,
followed by genuinely fresh-object evaluation. Spring-grid expansion is not
supported by this result.

Machine-readable metrics and provenance are in
`artifacts/source_summary.json`.
