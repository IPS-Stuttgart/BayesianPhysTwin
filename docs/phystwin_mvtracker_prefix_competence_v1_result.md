# PhysTwin MVTracker Prefix Competence v1 Result

## Decision

The frozen one-case competence gate **failed**. Stop the direct
MVTracker-to-PhysTwin route without tuning the tracker, cameras, sensor-depth
handling, scene normalization, visibility threshold, query identities,
anchoring, or acceptance thresholds.

This is an already-open, prefix-only source result. It is not evidence of
Bayesian-PhysTwin improvement, calibration, independent transfer, or state of
the art.

## Frozen Result

The predictor used only RGB-D frames `[90, 121)` from cameras 0, 1, and 2 of
`single_lift_cloth`. Benchmark identities 3, 4, 6, and 8 were initialized from
their frame-90 positions. The MVTracker prediction was sealed before the
withheld manual trajectories on those same prefix frames were scored.

| Metric or gate | Frozen result | Required | Pass |
| --- | ---: | ---: | :---: |
| Supported point-frames | 100.00% | at least 75% | yes |
| Identity RMSE | 18.209 mm | at most 15 mm | no |
| Exact-persistence RMSE | 34.516 mm | comparator | -- |
| Relative RMSE gain | 47.245% | at least 10% | yes |
| Last-six-frame RMSE | 23.407 mm | at most 15 mm | no |

The tracker substantially outperformed exact persistence, so the joint
multiview RGB-D estimate contains real motion information. It nevertheless
missed both absolute-accuracy gates, and its error increased at the end of the
prefix. The evidence therefore does not justify exposing a simulator
assimilation rule to this observation stream.

## Provenance

- Protocol lock commit:
  `2355d078a28cdb3b1a7e9f2ab8d146e8a21e81a3`.
- Locked protocol SHA-256:
  `5ca479ffe9ea4bad4ae7d58b2e70f5de64ad1ed81ebe0519fc2c27826d4ce0cb`.
- MVTracker revision:
  `ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072`.
- MVTracker checkpoint SHA-256:
  `a7fa86f2a7223e3e0aa4c1d3eff0dec5fe8a9227a48572ce943b8e49d8a4f8e6`.
- Prediction archive SHA-256:
  `fd533eaa3416c00e7c65687cd2f929cd90ec9a4072dd645d32509d5f255c17a4`.
- Prediction seal SHA-256:
  `0a267706de0a23ebaccb3d858131e86c2354c9eb2c23f1d4e17029e7396ecb95`.
- Result file SHA-256:
  `32c4efb68ba1fde3a522f79678ccf98af63134cd805f3c43172dd130c71eac27`.
- Internal canonical result SHA-256:
  `6d16551aaca766eef98b02965036ea5e44bf391b08f52eaaa66c09aa0476955b`.

Compact evidence is archived under
`results/sota/phystwin_mvtracker_prefix_competence_v1/`.

## Information Boundary

No RGB, depth, manual trajectory, point cloud, or simulator outcome at or
after frame 121 was read or scored. No held-v8, sealed PokeFlex, or PhysTwin
future artifact was accessed. Because the competence gate failed, no
assimilation smoke is authorized from this result.
