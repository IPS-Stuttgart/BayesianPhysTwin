# PhysTwin TAPNext++ Prefix Competence v1 Result

## Decision

The frozen one-case competence gate **failed** because observation support was
below the preregistered threshold. Stop this TAPNext++-to-PhysTwin route
without tuning visibility, mask, depth, reprojection, camera, query, support,
or gate choices. No assimilation run is authorized by this result.

This is an already-open prefix-only source result. It is not evidence of a
Bayesian-PhysTwin improvement, calibration, independent transfer, or state of
the art.

## Frozen Result

TAPNext++ used only causal RGB-D frames and object masks on `[68, 88)` from
the three released cameras of `single_lift_cloth`. Identities 3, 4, 6, and 8
were initialized from their frame-68 positions. The 3D prediction was sealed
before the withheld manual trajectories were opened.

| Metric or gate | Frozen result | Required | Pass |
| --- | ---: | ---: | :---: |
| Supported point-frames | 68.421% (52/76) | at least 75% | no |
| Identity RMSE | 5.090 mm | at most 15 mm | yes |
| Exact-persistence RMSE | 35.103 mm | comparator | -- |
| Relative RMSE gain | 85.499% | at least 10% | yes |
| Last-five-frame RMSE | 6.278 mm | at most 15 mm | yes |

The result isolates the bottleneck cleanly. On admitted rows, causal
TAPNext++ plus conservative metric lifting is substantially more accurate
than both the frozen absolute gate and exact persistence. The failure is
coverage: one camera is frequently occluded, and the two-view depth-consistent
rule still supports only 52 of 76 eligible material point-frames.

This is useful source evidence for future observation-provider design, but it
does not permit weakening the gate on this opened case. In particular, the
result does not show that sparse updates improve the simulator, that the
reported covariance is calibrated, or that coherent camera bias is absent.
Those questions require a newly locked independent evaluation and a
baseline-relative guarded belief update with exact fallback.

## Provenance

- Method prelock commit:
  `cd66090ff271764c8ea7d5c23cbfab5f19b85d97`.
- Source-artifact lock commit:
  `0018b4f`.
- Runtime-only CUDA amendment:
  `dc30eb7c3e370a65f4dba50e8eb1695a784cdb03`.
- Runtime-amendment lock commit:
  `590e573`.
- TAPNet revision:
  `c2cbab81cc06092b5f05bfe2da7bfec54e2079c9`.
- TAPNext++ checkpoint SHA-256:
  `6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f`.
- Source report SHA-256:
  `bc40c374d19b54054abaccf1d41089dbc2babf637acc9724717ea693db5cd4f0`.
- Prediction report SHA-256:
  `0e4a2c9fcd899d4561ed2a3eb4510bdc3c4c093e7d6f4089193dec857c37292c`.
- Prediction seal SHA-256:
  `a89dd9692719953996c800ac9cce90b1f974ea43de6a9270cb483a7def28c509`.
- Result file SHA-256:
  `09c808bccb23880cdef6ed21e8078af4104bbaad25eb1c17f61c7ca2be6d3a42`.
- Internal canonical result SHA-256:
  `240da82c5343dec803cd5d2f92e58d4bf540f39e1648ec42a0607a33cbadd34c`.

Compact evidence is archived under
`results/sota/phystwin_tapnextpp_prefix_competence_v1/`.

## Information Boundary

The prediction process never received the withheld manual prefix path. The
prediction was hashed and sealed before scoring. No observation at or after
frame 121, simulator future outcome, held-v8 artifact, or sealed PokeFlex
target was read or modified. Because the support gate failed, no guarded
assimilation smoke or larger cohort is authorized from this protocol.
