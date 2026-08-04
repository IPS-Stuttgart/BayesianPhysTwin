# PokeFlex Prior-Aware Belief Source v1

## Purpose

The existing PokeFlex experiments leave useful but non-transferable headroom.
Force- and action-supported Kinect corrections have a 7.72% oracle gain over
nine opened objects, while source-trained D405 regret guards fail on new object
classes. The failure is consistent with a common-mode ambiguity: a coherent
D405 displacement can be either object motion or sensor/model bias.

This source-only method tests the smallest corresponding Bayesian change. The
Kinect checkpoint and graph registration define a low-rank, physically
reachable state proposal. Independent D405 depth supplies the metric
likelihood. Shared and camera-centered translation terms are explicit nuisance
variables rather than state corrections.

## Statistical Boundary

The implementation keeps four quantities distinct:

1. Geometry determines soft material-point association probabilities.
2. Static D405 calibration determines prior perception reliability.
3. Assignment-mixture spread and D405 noise enter covariance in square metres.
4. The D405 innovation enters the grouped robust mixture likelihood exactly
   once.

The Kinect registration residual is proposal-only. It is not reused as
independent likelihood evidence. Dense rows from one D405 form one capped
correlation group, exact duplicate rows are removed, and an exact duplicate
camera panel is suppressed. The posterior state covariance is not allowed to
be tighter than the conditional known-bias reference computed with the same
final robust responsibilities.

## Causal Boundary

For source frame `s`, the released checkpoint predicts:

- state at `s` from Kinect frames `s-5` through `s-1`;
- state at `s+1` from Kinect frames `s-4` through `s`.

Kinect registration, robot transforms, force, and D405 depth are used only
through `s`. The candidate NPZ and its content-addressed seal are written before
the mesh at `s+1` is opened. A rejected numerical update returns the original
checkpoint vertex object exactly.

## Development Smoke

The first lock uses already-open `FoamDice_T1`. The source frame is selected by
an outcome-blind rule: the earliest frame at or after six with measured force-y
above 3 N and an available next frame. Four force/action-reachable fields at a
0.25 object-relative support radius form the state span. No coefficient, rank,
prior variance, association threshold, or trust-region setting may be changed
from the one-frame outcome.

The smoke is a technical and directional check only. A favorable result can
justify freezing a separate object-held-out source protocol over the already
opened PokeFlex source objects. It cannot authorize opening calibration or
target objects, and it cannot support a state-of-the-art claim.

## Smoke Result

The frozen smoke completed on source frame 7 and predicted frame 8. Kinect
registration was admissible, with 797 associated points and a 5.86 mm RMS
proposal update. The D405 belief retained 512 clustered rows in two capped
sensor groups. Static calibration assigned prior reliabilities of 0.05 and
0.926 to the two groups; the lower value reflects a 27.33 mm calibration p90
residual in the first camera rather than the current state innovation.

The unconstrained fixed-point solution implied a maximum 4.20 mm query update.
The released checkpoint's one-step physical response admitted only 2.58 mm
under the locked two-times-response bound. Inference therefore returned
`implausible-state-update`, and the selected prediction was the released
checkpoint exactly:

| Method | CD_UL1 at frame 8 | Relative change |
| --- | ---: | ---: |
| Released checkpoint | 4.427 mm | reference |
| Prior-aware selected result | 4.427 mm | 0.00% |

This is a successful exact-fallback check but supplies no accuracy evidence.
It does not justify loosening the trust region. Later source frames have larger
physical responses, so a source-panel run with the identical frozen method is
the next diagnostic; calibration and target objects remain sealed.

The source panel is locked in
`configs/sota/pokeflex_prior_aware_belief_source_panel_v1.json`. It covers the
five previously opened source objects and takes `T1`, `T3`, `T4`, `T5`, and
`T6`; `T3` remains restricted to its historical 40-frame design interval. Each
candidate uses D405 only at `f-1`, is content-sealed before mesh `f` is read,
and retains the smoke's method bytes and parameters. Advancement requires at
least 1% object-balanced improvement, four object wins with no losses, at most
1% regression on any object, at most 10% harmful admitted frames, and at least
5% numerical admission. These are source-development gates only.

Evidence:

- pre-outcome implementation commit: `6f708e5b3eff840df810d4575c7c560f889a8498`;
- result SHA-256: `07530a1c7e30af2b609aa0db3d03f24806039f4b306e9e9f3315ac1cda8d800e`;
- prediction-seal SHA-256: `92cd0f2eb5a249b956a6edddd0092bfafae5290b7f8d742c659213495139cae1`;
- server run root:
  `/mnt/corsair/florianpfaff/pokeflex-prior-aware-source-smoke-v1-6f708e5`.

## Relationship to Causal4D

This is a Bayesian-PhysTwin observation/state-belief experiment. It neither
changes Causal4D's frozen intervention-abduction claim nor uses Causal4D target
artifacts. The resulting narrow observation artifact may later be consumed by
Causal4D only after independent source and calibration gates pass.
