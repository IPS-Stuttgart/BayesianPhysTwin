# PokeFlex Dual-Sensor Regret Kill Test

## Question

The failed independent-object D405 guard was followed by one post-open kill
test. A candidate generated for frame `f-1` was scored when frame `f-1` became
observable by both Kinect cameras and both independently mounted D405 cameras.
An arm with the same fixed radius and scale was proposed for frame `f` only if
the worst regret in both sensor families was negative.

This test used only the four already-open calibration objects. It is not a
prospective result, and the eight target objects remained sealed.

## Result

Pure delayed consensus failed:

| Method | Object-balanced CD_UL1 | Relative change | Object wins | Object losses |
| --- | ---: | ---: | ---: | ---: |
| Released checkpoint | 4.817 mm | reference | - | - |
| Dual-sensor consensus | 4.963 mm | **+3.04%** | 2/4 | 2/4 |

The gate accepted 158 of 276 frames. Of those, 102 improved and 56 regressed.
`3dPrintedPyramid` worsened by 44.09%, showing that delayed Kinect and D405 can
agree on the same geometrically wrong update.

Using delayed Kinect only as a veto on the earlier source-UCB selector also
failed. A zero-margin veto rejected none of its 71 accepted frames; increasing
the margin removed useful and harmful updates together. At a 2 mm veto margin,
only one harmful update survived and aggregate improvement was still negative.

## Decision

Stop treating another camera-family fit as independent safety evidence. The
result is consistent with the common-mode-bias ambiguity already demonstrated
on Deform360. The next candidate must change the physical support of the state
update or add a genuinely non-camera-equivalent observation, rather than add
another confidence rule over the same geometric residual.

The next bounded hypothesis combines measured force/action reachability with
the independent D405 anchor. It will first be tested on opened development data;
no target object is authorized by this result.

## Evidence

- Evaluation:
  `results/sota/pokeflex_dual_sensor_regret_kill_v1/calibration_consensus.json`
- Evaluation SHA-256:
  `a5a29b6749143d18a147d1f7414855ddbdb590221817866996fcb6566cc12fd9`
- Implementation commit: `2d66cb1`
