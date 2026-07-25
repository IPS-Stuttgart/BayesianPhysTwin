# PokeFlex Regret-Guard Calibration v1

## Purpose

The source-calibrated D405 regret guard passed its registered same-object/new-take
replication: 6.418 to 6.222 mm CD_UL1, a 3.06% improvement, with both objects and
all three takes improving. This protocol asks whether the exact same deployment
artifact transfers to previously unseen objects.

No coefficient, feature, support range, selector correction, candidate arm, or
threshold is refit on calibration data.

## Cohort

The four calibration takes were fixed in the original PokeFlex split and exist
in the public download:

- `3dPrintedPyramid_T2`;
- `Beanbag_T2`;
- `FoamCylinder_T2`;
- `PlushMoon_T2`.

Only archive names and sizes were checked before lock. No archive member, sensor
frame, mesh, or outcome was read. There is no replacement. The eight target
objects remain sealed.

## Gates

The no-refit calibration result passes only if all conditions hold:

- at least 1% object-balanced improvement over the released checkpoint;
- at least three of four object means improve;
- no object regresses by more than 1%;
- at most 10% of accepted frames are harmful;
- at least one frame is accepted.

The published 6.498 mm Kinect reference is reported but is not a gate because
this calibration cohort is not the official target cohort.

Passing allows a target protocol to be drafted. It does not open target objects
automatically and is not itself a state-of-the-art claim.

## Result

The frozen deployment failed the independent-object gate. The selector was
applied once, without refitting or recalibration, after all four candidate banks
had been generated.

| Method | Object-balanced CD_UL1 | Relative change | Object wins | Object losses |
| --- | ---: | ---: | ---: | ---: |
| Released checkpoint | 4.817 mm | reference | - | - |
| Regret-guarded update | 4.872 mm | **+1.16%** | 2/4 | 1/4 |

The guard accepted 71 of 276 target frames. Of those accepted frames, 43
improved and 28 regressed, giving a 39.44% false-safe rate. It returned the
checkpoint exactly on the other 205 frames.

| Object | Baseline | Guarded | Relative improvement |
| --- | ---: | ---: | ---: |
| `3dPrintedPyramid` | 2.369 mm | 2.827 mm | -19.36% |
| `Beanbag` | 4.876 mm | 4.876 mm | 0.00% |
| `FoamCylinder` | 4.432 mm | 4.345 mm | 1.98% |
| `PlushMoon` | 7.589 mm | 7.441 mm | 1.96% |

Only the nonempty-acceptance gate passed. The object-balanced improvement,
object-win, maximum-regression, and false-safe gates all failed. The selected
mean remains below the published 6.498 mm reference, but that comparison is
report-only because this is a different four-object cohort.

This result closes the present guard without opening the eight target objects.
The positive same-object replication did not generalize safely to independent
objects. No target protocol may be drafted from this deployment, and the
calibration outcomes must not be used to tune a successor claimed on this
cohort.

The scientific implication is narrower and useful: independent depth can help
identify candidate regret, but the source-calibrated regret mapping is not
object invariant. A successor needs an object-conditional or genuinely
distribution-free safety certificate validated on fresh objects; simply
retuning this regression is not a credible route to a target claim.

## Lock

- Protocol:
  `configs/sota/pokeflex_independent_depth_regret_guard_calibration_v1.json`
- Protocol SHA-256:
  `cef28df26d7710670aa3d73883462e896ec2d903c3be5faf00fee6a8d00a2644`
- Frozen source result SHA-256:
  `6065d1178796eba949b0411fbd57b53184e39d38f6559357c740d63b8b47398b`
- Parent prospective result SHA-256:
  `0fc409a1bf85d4ef0e697bdb5604689094914ac57d32652660685007bfd67b98`
- Calibration evaluation:
  `results/sota/pokeflex_independent_depth_regret_guard_calibration_v1/calibration_evaluation.json`
- Calibration evaluation SHA-256:
  `63cdd6fd06419937a7515356e1cb9f3b9177d879a1047071f3689f85d4605e93`
