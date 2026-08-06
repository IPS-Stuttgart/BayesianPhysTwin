# TAPNext++ Frame-Zero Material Transport Provider Result

Date: 2026-08-06

## Result

The frozen 14-case opened-source provider gate passed every preregistered
criterion:

| Quantity | Result |
| --- | ---: |
| evaluated cases | 14/14 |
| cases passing every per-case gate | 14/14 |
| completed row support | 981/989 (99.19%) |
| case-balanced identity RMSE | 5.006 mm |
| case-balanced gain over exact persistence | 74.18% |
| technical failures or replacements | 0 |

The aggregate decision is
`authorize-locked-material-transport-assimilation-source-study`. This
authorizes only a separately checksummed source assimilation experiment. It
does not establish that the observations improve the untouched physical
future.

## Per-Case Provider Evidence

| Case | Support | Candidate RMSE (mm) | Persistence RMSE (mm) | Gain |
| --- | ---: | ---: | ---: | ---: |
| `double_lift_cloth_3` | 89.33% | 9.379 | 12.323 | 23.89% |
| `double_lift_sloth` | 100.00% | 4.211 | 41.362 | 89.82% |
| `double_stretch_zebra` | 100.00% | 4.255 | 67.893 | 93.73% |
| `single_clift_cloth_1` | 100.00% | 5.481 | 8.846 | 38.04% |
| `single_clift_cloth_3` | 100.00% | 2.674 | 14.333 | 81.35% |
| `single_lift_cloth` | 100.00% | 3.581 | 18.143 | 80.26% |
| `single_lift_cloth_1` | 100.00% | 5.784 | 20.435 | 71.70% |
| `single_lift_cloth_3` | 100.00% | 4.892 | 16.958 | 71.15% |
| `single_lift_cloth_4` | 100.00% | 6.269 | 17.019 | 63.17% |
| `single_lift_sloth` | 100.00% | 2.813 | 29.016 | 90.31% |
| `single_lift_zebra` | 100.00% | 4.069 | 32.660 | 87.54% |
| `single_push_rope` | 100.00% | 6.491 | 32.468 | 80.01% |
| `single_push_rope_1` | 100.00% | 4.604 | 21.276 | 78.36% |
| `single_push_sloth` | 100.00% | 5.582 | 51.676 | 89.20% |

## What Changed Relative to the Failed Assimilation Arm

The earlier sparse assimilation associated every observation to graph nodes
from the current source-frame geometry. A post-open audit found that only
2/27 geometry-MAP nodes matched the benchmark's fixed frame-zero material
identity. Accurate tracker positions could therefore update the wrong
material nodes.

This study fixes one graph-node identity at frame zero and evaluates material
transport over the terminal 20-frame training prefix. The future
assimilation uses relative displacement,

```text
(observed position at t - observed position at frame zero)
- (physical fixed-node position at t - physical fixed-node position at frame zero),
```

so a static query-to-node offset cancels. Its magnitude is retained as metric
covariance rather than silently treated as certainty. The query frame is an
anchor and is not itself an update row.

## Custody and Provenance

- implementation commit: `d74f6597`
- source-lock commit: `c6e1539e`
- protocol SHA-256:
  `b4e65b8f073fdda855c3d35d153d8920e61a8ae32237f95621ed90f6ed0d9a08`
- source manifest canonical SHA-256:
  `a3443cc082891c978a740a41981e0b3febd1982043548cff9e27868155b28e8f`
- aggregate summary file SHA-256:
  `39ac705f49e60219e2fc6e78e31d60db56ce9bb8c719fd4d2745a8ebafd8ca96`
- aggregate canonical result SHA-256:
  `df5babfc29266a5da9952b5ddb2fb3b148dcf9b33da5c55c1dbde01639bf2637`
- TAPNext++ revision:
  `c2cbab81cc06092b5f05bfe2da7bfec54e2079c9`
- TAPNext++ checkpoint SHA-256:
  `6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f`

All 14 strict predictions and all 14 completed carriers were sealed before
the withheld prefix identity targets were opened. No future simulator outcome
or held-v8 artifact was read. The initial partial-root staging failure remains
preserved in the source lock and produced no prediction or score.

## Claim Boundary

This is a source-side material-identity bridge result on already-open
PhysTwin data. Manual benchmark identity initializes the query and fixes the
frame-zero graph node. It is neither deployable automatic association nor an
independent state-of-the-art result. The next legitimate question is whether
the frozen bridge improves disjoint hidden identities and geometry in the
untouched future beyond dense persistence.
