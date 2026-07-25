# PokeFlex Independent-Depth Source Validation v2

## Scope

This study asks whether the eye-in-hand RealSense D405 pair can guard a
Kinect-derived Bayesian state update using evidence from a separately mounted
metric-depth family. It is a Bayesian-PhysTwin observation/state-update study,
not a Causal4D result, a target-object result, or a claim that Kinect and
RealSense errors are statistically independent.

The camera-only Deform360 studies established an identifiability limit under a
shared coherent bias. PokeFlex offers a useful next test because the released
Kinect predictor and the eye-in-hand RealSense pair do not use the same camera
family. They can still share world-frame or calibration error, so the two D405
views are combined by the worst qualified-sensor regret rather than by
independent precision addition.

## Causal Boundary

For target frame `f`:

- the Kinect checkpoint consumes frames `f-5` through `f-1`;
- the Kinect graph update, force/pose history, and RealSense evidence stop at
  `f-1`;
- the RealSense regret observed at `f-1` selects the same named update arm for
  `f`;
- frame `f` Kinect/RealSense data and the frame `f` volumetric mesh are not used
  before prediction;
- the volumetric mesh is used only for scoring.

The static template mesh is allowed for translation calibration and a fixed
15 mm material-support neighborhood. This association step does not use the
candidate innovation or future outcome as prior reliability.

## Frozen Method

The source-validation protocol is
`configs/sota/pokeflex_independent_depth_source_validation_v2.json`, with
canonical payload hash
`04ecf64f4b15e543825732b4f9296119ebd5fe47d99dcfc8230cfb70f0c86051`.

The selector uses:

- robust translation-only D405-to-template calibration;
- exclusion of a sensor whose static median residual exceeds 10 mm;
- exact checkpoint fallback if no sensor remains;
- a 15 mm static-template support neighborhood;
- fixed information mass after 4 mm voxel clustering;
- maximum baseline-relative regret over qualified D405 sensors;
- the frozen three-radius, five-scale action-local update bank;
- a one-frame-lag arm policy and zero acceptance margin;
- byte-exact checkpoint fallback when no arm has negative predicted regret.

`T3` was the method-design take. Full trajectories from `T1`, `T4`, `T5`, and
`T6` were then evaluated once under the fixed v2 method. `T2`, all calibration
objects, and all target objects remained sealed.

The previous runner remains byte-identical at SHA-256
`79ba8946653a55a70dc0b990e874754397e18948b9b7ba541158c6641cfc4b43`.
The independent-depth path lives in a separate runner.

## Results

| Evidence | Baseline CD_UL1 | Guarded CD_UL1 | Relative improvement | Object wins | Object losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| T3 design smoke | 5.846 mm | 5.536 mm | 5.31% | 5/5 | 0/5 |
| Frozen full source validation | 4.649 mm | 4.466 mm | 3.93% | 3/5 | 2/5 |

The full source-validation object means were:

| Object | Baseline CD_UL1 | Guarded CD_UL1 | Relative improvement |
| --- | ---: | ---: | ---: |
| `3dPrintedHeart` | 3.695 mm | 3.757 mm | -1.68% |
| `FoamDice` | 6.034 mm | 5.454 mm | 9.62% |
| `MemoryFoam` | 2.350 mm | 2.360 mm | -0.42% |
| `PlushOctopus` | 5.585 mm | 5.517 mm | 1.21% |
| `ToiletPaperRoll` | 5.581 mm | 5.243 mm | 6.06% |

Independent-depth regret remained informative over 15,816 candidate-frame
pairs:

| Diagnostic | Value | Registered gate | Passed |
| --- | ---: | ---: | :---: |
| Spearman regret correlation | 0.558 | at least 0.20 | yes |
| Regret sign agreement | 68.49% | at least 65% | yes |
| False-safe rate among accepted arms | 9.62% | at most 10% | yes |
| Object-balanced CD improvement | 3.93% | at least 5% | **no** |
| Object wins | 3/5 | at least 4/5 | **no** |
| Maximum object regression | 1.68% | at most 10% | yes |

The registered gate therefore failed and `T2` access is not permitted.

## Interpretation

The positive part is real: an independently mounted depth family predicts
hidden candidate regret far better than the earlier camera-only selectors and
supports useful updates on three material/shape families. The 15 mm support
filter also converts the original `MemoryFoam` smoke regression from about 43%
to conservative near-fallback behavior.

The limitation is equally clear. The D405 observation measures whether an arm
fit the previous frame, while v2 transfers that arm identity to a newly inferred
correction at the next frame. That temporal arm policy is unstable across
materials, especially for `3dPrintedHeart`, even though the independent sensor
itself remains correlated with hidden regret.

The next source-only hypothesis is therefore a same-time Bayesian state update:

1. at frame `f-1`, form the current action-supported source-state candidates;
2. score those current candidates with qualified D405 evidence from `f-1`;
3. infer a guarded posterior or shrinkage over the current correction magnitude;
4. propagate that selected state correction once through the physical/action
   model to frame `f`;
5. retain a latent shared sensor bias, worst-sensor aggregation, and exact
   fallback.

This removes the avoidable one-frame arm-transfer assumption while preserving
the causal boundary. It must be developed and cross-validated only on the
already opened source cohort. `T2` stays sealed unless a new protocol is frozen
and its source gates pass.

## Evidence

- Design summary:
  `results/sota/pokeflex_independent_depth_source_validation_v2/design_summary.json`
- Frozen source result:
  `results/sota/pokeflex_independent_depth_source_validation_v2/summary.json`
- Checksummed execution manifest:
  `results/sota/pokeflex_independent_depth_source_validation_v2/execution_manifest.json`

The repository summaries contain no raw PokeFlex frames or meshes. Full
frame-level records remain in the server evidence directory and are referenced
by SHA-256 from the compact result.
