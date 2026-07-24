# PokeFlex Robot-Checkpoint Fusion Source Result

## Status

**Source gate failed. Do not inspect the opened calibration objects or sealed
target objects for this method.**

This study asks whether the official PokeFlex force-and-tool-only reconstruction
checkpoint provides independent evidence that safely improves the stronger
official Kinect point-cloud checkpoint. It is a source-only observation-fusion
study, not a material-parameter, Causal4D, or state-of-the-art claim.

The frozen protocol is
`configs/sota/pokeflex_robot_fusion_source_v1.json`, with canonical protocol
checksum
`e1ca97dfab720f427561f40c1eac3958fc00edbd65319bb74a8c5302621900ce`.
The evaluation contains 1,418 causal target frames from 20 takes and five
already-open development objects. Every prediction uses frames `f-5` through
`f-1`; target geometry is loaded only after candidate construction.

## Method

The official robot checkpoint consumes five measured wrench/tool-position
records and predicts the same template topology as the official Kinect
checkpoint. The frozen candidate bank interpolates from the Kinect prediction
toward that independent proposal:

```text
x_candidate = x_kinect + scale * (x_robot - x_kinect)
scale in {0, 0.05, 0.1, 0.2}
```

Scale zero returns the Kinect vertices byte-for-byte. A fixed blend is only a
diagnostic. The admissible method is a leave-one-object-out, source-calibrated
upper regret bound using target-independent force, tool-motion, deformation,
and model-disagreement features. Candidate selection receives a second
selector-aware regret correction and falls back exactly when unsupported.

## Results

| Arm | Object-balanced CD UL1 | Relative change | Object wins | Worst object regression |
|---|---:|---:|---:|---:|
| Kinect baseline | 4.649 mm | 0.00% | 0/5 | 0.00% |
| Fixed scale 0.05 | 4.584 mm | +1.40% | 5/5 | 0.00% |
| Fixed scale 0.10 | 4.564 mm | +1.84% | 3/5 | 1.38% |
| Fixed scale 0.20 | 4.632 mm | +0.36% | 1/5 | 13.87% |
| Per-frame oracle | 4.412 mm | +5.09% | diagnostic only | diagnostic only |
| LOO regret guard | 4.639 mm | **+0.23%** | **2/5** | 0.027% |

The LOO guard accepted 22 of 1,418 frames. Six accepted frames regressed,
giving a 27.27% false-safe rate. Its candidate upper-bound coverage was 92.41%.

The frozen source gates require at least 5% object-balanced improvement, four
object wins, no more than 10% per-object regression, and at most 10% false-safe
acceptance. The result passes only the maximum-regression gate:

```text
object-balanced improvement: FAIL
object wins:                 FAIL
maximum object regression:  PASS
false-safe rate:             FAIL
nonempty acceptance:         PASS
```

The complete result and source-artifact hashes are in
`results/sota/pokeflex_robot_fusion_source_v1/source_cross_object_evaluation.json`.

## Interpretation

The independent robot channel contains a small correction signal: the most
conservative fixed blend improves all five development objects. It is not
strong enough for safe frame-level admission. The frozen bank's 5.09% oracle
headroom is essentially equal to the required transfer threshold, leaving no
room for uncertainty, source-to-object transfer, or selector error.

This result does not show that wrench/tool evidence is useless. It rejects this
specific use of the released robot checkpoint as a convex full-mesh correction
to the released Kinect reconstruction. The checkpoint adapter remains useful
as a versioned independent-modality feeder or diagnostic, particularly for
common-mode camera-bias studies.

## Decision

1. Do not open the four calibration objects for this arm.
2. Do not author a sealed-target protocol from this candidate family.
3. Do not tune scales, features, or regret thresholds on these outcomes.
4. Preserve the exact-fallback adapter and source evidence as a negative result.
5. Require materially larger source oracle headroom before revisiting
   force/tool fusion.

The strongest Bayesian-PhysTwin path remains a guarded state/discrepancy belief
update with independent evidence and exact fallback, rather than a fixed blend
between reconstruction networks.
