# PokeFlex action-conditioned discrepancy diagnostic v1

## Status

This is a post-open source diagnostic, not a prospective result. It uses 24
already-open takes from five development objects and four previously opened
calibration objects. All eight PokeFlex target objects remained sealed. The
diagnostic gate failed, so this result does not authorize a target run.

## Question

The Cloth Sim2Real action-phase diagnostic found transferable headroom for a
low-capacity translation correction. This experiment asks whether a comparable
correction transfers across PokeFlex objects when it is predicted only from
causal robot and physical-checkpoint features.

For target frame `f`, the feature vector uses only:

- force, tool-pose, and end-effector histories from frames `f-5` through `f-1`;
- the released PokeFlex checkpoint deformation predicted from those same
  frames.

The feature construction is independent of the target residual. Target
geometry is loaded only afterward to form the post-open diagnostic label and
score. A robust global-translation label retains the closest 90% of
prediction-to-target nearest-neighbor matches. The learned translation is
capped at 10 mm.

The model is a standardized ridge regression fitted leave-one-physical-object
out with equal object weight. A constant-only regression is evaluated through
the identical folds as the essential control. Both arms use the fixed scale
bank `{0, 0.25, 0.5, 0.75, 1}`; scale zero is an exact fallback. Choosing the
best scale after opening these outcomes is explicitly diagnostic.

## Result

The metric is unidirectional L1 Chamfer distance in millimeters, aggregated
equally over frames within take, takes within object, and objects.

| Arm | Object-balanced error (mm) | Improvement vs baseline | Object wins | Worst object change |
| --- | ---: | ---: | ---: | ---: |
| Released checkpoint baseline | 4.7242 | 0.00% | - | - |
| Per-frame translation oracle | 4.4445 | 5.92% | - | - |
| Constant translation, best post-open scale 1 | 4.6729 | 1.09% | 8/9 | -0.35% |
| Causal action-conditioned translation, best post-open scale 1 | 4.6669 | 1.21% | 7/9 | -0.58% |

The causal features reduce error by only 0.13 percentage points more than the
constant control. The per-frame oracle itself exposes only 5.92% object-balanced
headroom, leaving almost no margin for a causal cross-object predictor.

The registered source gate required all of:

- at least 5% object-balanced improvement;
- at least 7/9 object wins;
- no object regression greater than 10%.

Both learned arms pass the win and maximum-regression checks but fail the 5%
improvement check.

## Interpretation

This closes the tested family of bounded global translations predicted from the
five-frame force/tool/end-effector history and released-checkpoint deformation
on PokeFlex. The near-equivalence of the causal and constant arms shows that
these action features do not explain a useful transferable portion of the
remaining global offset. More importantly, the small oracle ceiling says this
is not primarily a learner-capacity failure.

No larger PokeFlex evaluation is justified for this model family, and the
sealed target cohort must remain unopened. This result does not close
spatially varying, contact-local, or belief-state corrections. The primary
prospective route remains the separately frozen held-v8.3 guarded online-belief
qualification.

## Evidence

- Causal artifact:
  `results/sota/pokeflex_action_discrepancy_opened_v1/causal.json`
  (`b253d04572902e8571f6325d787f144e9168f0980fd36fa74db0fd6f53517035`)
- Constant control:
  `results/sota/pokeflex_action_discrepancy_opened_v1/constant.json`
  (`3d8abedacb0462bdde7ad75eec50e3001913a58fbc867a1f5704aa27ffb1fe3f`)
- Evidence size: 24 takes, 9 objects, and 1,694 target-frame rows.
- Sealed target object access recorded by both artifacts: `false`.
