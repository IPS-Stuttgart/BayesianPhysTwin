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

## Source-Panel Result

The frozen 25-take source panel completed all 1,479 registered active target
frames without replacement or technical failure. The fixed posterior admitted
900 frames (60.85%), but 474 admissions were harmful. It therefore failed the
accuracy, object-transfer, worst-object, and false-safe gates.

| Object | Released checkpoint | Prior-aware selected | Relative improvement | Selective oracle |
| --- | ---: | ---: | ---: | ---: |
| `3dPrintedHeart` | 4.698 mm | 4.739 mm | -0.89% | +0.69% |
| `FoamDice` | 5.725 mm | 5.739 mm | -0.24% | +0.60% |
| `MemoryFoam` | 2.413 mm | 2.473 mm | -2.47% | +0.31% |
| `PlushOctopus` | 5.372 mm | 5.354 mm | +0.34% | +0.70% |
| `ToiletPaperRoll` | 6.244 mm | 6.245 mm | -0.03% | +0.48% |
| **Object-balanced** | **4.890 mm** | **4.910 mm** | **-0.40%** | **+0.58%** |

Only one of five object means improved, the worst object regressed by 2.47%,
and the false-safe rate among admitted frames was 52.67%. The post-open
per-frame accept/reject oracle reaches only 0.58% object-balanced improvement;
every object remains below 0.71%. This ceiling is already below the registered
1% source-advancement threshold, before estimating any selector.

The nearest-surface covariance diagnostic reports 86.3% to 100.0% nominal 90%
coverage across objects, but it includes a fixed 4 mm readout floor and does not
provide material identities. It is therefore a diagnostic, not a calibration
claim. Together with the negative mean, it indicates that wide uncertainty did
not repair the candidate's weak and object-dependent point estimate.

The result closes this candidate family. Explicit shared and centered-view
translation nuisances prevent a simple camera gauge from being forced entirely
into state, but they do not identify object-dependent, non-rigid D405 bias well
enough to improve the checkpoint. No source-trained regret guard is justified:
even its perfect selective oracle lacks the registered headroom. Calibration
and target objects remain unauthorized and unopened for this method.

Panel evidence:

- pre-outcome panel commit: `fcc36509333ad469665a91781cf01314c020b0e3`;
- summary SHA-256: `d93889b9869f34f6b81c9ce9a8fe475b9f1f65bd5e208b21396b834ad7fea63d`;
- progress SHA-256: `ba4d3f3a72b710c16ef2f39917cd889caec96bd2b0580a7ad6315eb4f79cad4b`;
- post-open oracle audit SHA-256:
  `7a5adc61b5d9dffb3f6a448820a3fa0d096c292d4331281ce7d9cb21c7e10d1f`;
- take summaries SHA-256:
  `b7344e06f9a3bcdcb6a05448f022ded00e27d49d3a2d88f3512deb9c61870b27`;
- server run root:
  `/mnt/corsair/florianpfaff/pokeflex-prior-aware-source-panel-v1-fcc3650`.

## Relationship to Causal4D

This is a Bayesian-PhysTwin observation/state-belief experiment. It neither
changes Causal4D's frozen intervention-abduction claim nor uses Causal4D target
artifacts. The resulting narrow observation artifact may later be consumed by
Causal4D only after independent source and calibration gates pass.
