# DLO-Lab unknotting headroom development v1 result

## Decision

The frozen public-simulator screen terminated at native qualification on the
first of nine registered worlds. The exact action bank is rejected and no retry,
action-headroom analysis, source transfer, or prospective evaluation is
authorized.

This is a query-specific negative result, not evidence that DLO-Lab cannot model
unknotting.

## Frozen outcome

| Item | Result |
| --- | ---: |
| Attempted / completed worlds | 1 / 1 |
| Unrun worlds | 8 |
| Failed native check | segment length |
| Frozen maximum relative segment error | 10.00% |
| Reported maximum during rollout | 18.11% |
| Independently reconstructed final-state maximum | 15.75% |
| Attachment-offset drift | 0.00022 mm |
| Duplicate reward difference | 0 |
| Reward reconstruction error | 1.24e-9 |
| Development value analysis | not performed |

The run otherwise completed normally. All eleven native members had finite final
rewards in [0, 1], the two duplicate trajectories were exact, the rotation and
common-prefix checks passed, and the sealed final reward was reconstructed from
the final rope coordinates. The write-once result ID is
`ad50ae325285894aa6b108d17b2d6d76faee95228d656397725a2b6935c68a07`.

The final-state segment check was independently recomputed against the released
`ropec.npy` rest geometry. It confirms a 15.75% deviation even without relying on
the runner's maximum-over-time measurement. Arithmetic verification is not
independent human review.

## Interpretation

The prospective attachment-offset definition worked as intended: its maximum
drift was far below the 1 mm limit. The registered action family instead violated
the frozen geometry-preservation criterion. Because qualification failed before
the complete world bank existed, the observed first-world action rewards are not
used to select or redesign an action in this protocol.

The incumbent fixed-action fallback remains unchanged. Any future unknotting
study must be a separately motivated protocol, not a threshold relaxation or
retry of this one.

## Boundary

The run used the public DLO-Lab simulator on CPU. It read no protected target,
held-v8, DLO4, DLO5, or newly recorded data, and performed no official DLO3
evaluation.
