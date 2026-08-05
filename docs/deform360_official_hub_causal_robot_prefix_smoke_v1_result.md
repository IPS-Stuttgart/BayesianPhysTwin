# Deform360 causal all-camera robot-prefix smoke v1

## Result

The frozen source-only technical gate passed for the lexicographic calibration
smoke case. The causal all-calibrated-camera estimator recovered both UMI
gripper trajectories from frames `[108, 150)` and stopped before the untouched
future. No prediction score, confirmation payload, target outcome, or held-v8
artifact was opened.

| Quantity | Result |
|---|---:|
| Calibrated cameras | 36 |
| Prefix frames | 42 |
| Direct wrist support, grippers 0 / 1 | 100% / 100% |
| Both-finger support, grippers 0 / 1 | 73.81% / 97.62% |
| Contact-tail ready frames, grippers 0 / 1 | 6 / 6 |
| Observed opening range | 44.37-94.79 mm |
| Released UMI opening range | 40-112 mm |
| Maximum translation step | 2.112 mm |
| Maximum rotation step | 0.961 degrees |
| Admission | pass |

The result is not a comparison against prediction outcomes. It establishes only
that an independent metric gripper/contact coordinate source can be recovered
causally and reproducibly for this case.

## Camera-policy control

The earlier technical pass reused the three-camera object-motion panel and
produced a maximum gripper opening of about 416.9 mm, far outside the released
112 mm UMI maximum. The all-camera result demonstrates that the visual-provider
panel and the proprioception panel cannot be assumed interchangeable. The
three-camera estimate remains rejected and is not used downstream.

## Frozen execution

- Lock ID: `7e4f7a30d9ad00da9f47d2c0debd42fea704c0985e65f78f8cd4f584dc52bc34`
- Estimator implementation: `2b55280ecbccf77dfeafde3ba86191294c162670`
- Runtime revision: `1b83ba49aed6e442e9c73c5d2d4002ca7c2fdd56`
- Upstream Deform360 revision: `d8522a4403b766aeb387510c04e89032a56fdf35`
- Artifact ID: `555943b0614e211e9c4d0fb5e2928c5499d695f5d6a020b6d26c2646d536297e`
- NPZ SHA-256: `eed8e713d3159ec110002cb0281544c3afed350d8e657ac645174a981799d802`
- Manifest SHA-256: `95256bbfffc11268b1c9044b9a1f0b76616221a39e707fe414674923906cd6d0`
- Bound source files: 110

An independent second execution reproduced the NPZ and manifest byte for byte.

## Decision

The gate authorizes a source-only tactile-contact geometry feasibility artifact.
It does not yet authorize a displacement anchor or calibration score. The next
artifact must preserve the unresolved mapping between `tactilel`/`tactiler` and
the two recovered grippers as an explicit assignment mixture. It may associate
active taxel locations with object hypotheses, but visual residual magnitude
must not be reused as prior reliability.
