# MatPhys causal-prefix development result

## Question

This development run asked whether the pinned public MatPhys architecture could
improve released PhysTwin held-future prediction when every RGB frame and every
physical training objective was restricted to the released prefix. The run used
the already examined `double_lift_zebra` case and a global one-part material
proxy, so it was a model-family gate rather than independent confirmation or a
reproduction of MatPhys's part-level feed-forward claim.

The protocol and progression rule were frozen in
[`configs/sota/matphys_causal_prefix_200_dev_v1.json`](../configs/sota/matphys_causal_prefix_200_dev_v1.json)
before the official future metrics were read.

## Information audit

- Training and checkpoint selection used frames 0 through 39 only.
- The held-future evaluation starts at frame 40 and ends at frame 57.
- The maximum accessed RGB frame and maximum objective frame were both 39.
- Checkpoint selection used the fixed terminal epoch, not future validation.
- The model and optimizer states were finite.
- The known future controller trajectory was available, matching the released
  PhysTwin prediction setting.

The terminal checkpoint SHA-256 is
`192307f71d39b2160b8066f3fda0359a5eb68616a3ca51d9f8992428eaeef3be`.

## Result

| Official held-future metric | Released PhysTwin | Candidate | Relative change | 5% gate |
|---|---:|---:|---:|---:|
| Chamfer distance | 14.230 mm | 13.498 mm | -5.15% | Pass |
| Manual-track error | 25.877 mm | 27.014 mm | +4.39% | Fail |

The candidate narrowly clears the Chamfer threshold of 13.519 mm but misses the
manual-track threshold of 24.583 mm and instead regresses. Because both metrics
were required to improve by at least 5%, the overall gate fails.

## Decision

The registered action is to stop this per-case causal model family and not run
the proposed multi-case continuation. The result suggests that this coarse
global spring parameterization can improve surface agreement without improving
the identity-sensitive manual tracks that matter for deformable state
prediction.

This negative result does not invalidate the separate public-code all-frame
reconstruction control. That control uses future observations and answers a
different, transductive question. It also does not justify tuning this causal
run against the opened future. The next causal model should introduce a
different source-trained or partial-belief mechanism under a separately frozen
development protocol.

The compact machine-readable record is
[`results/sota/matphys_causal_prefix_200_dev_v1_result.json`](../results/sota/matphys_causal_prefix_200_dev_v1_result.json).
