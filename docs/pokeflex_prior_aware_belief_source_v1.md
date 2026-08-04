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

## Relationship to Causal4D

This is a Bayesian-PhysTwin observation/state-belief experiment. It neither
changes Causal4D's frozen intervention-abduction claim nor uses Causal4D target
artifacts. The resulting narrow observation artifact may later be consumed by
Causal4D only after independent source and calibration gates pass.
