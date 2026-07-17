# Bayesian residual reusable twin: source protocol v1

## Motivation

The current evidence rejects two shortcuts:

- the official Warp parameter grid is 50.6% worse than persistence on the
  penguin source actions;
- a leave-one-action-out rank-4 graph-spectral ridge is 3.21% worse overall
  and 6.58% worse late, with a large transfer failure on episode 6.

The per-episode contact-onset oracle retains 5.67% mean headroom, but its timing,
length scale, and gain vary sharply by action. A shared timing or linear-response
patch is therefore not the next candidate.

PGRD, released on 2026-07-15, already combines an optimized spring-mass model
with Point Transformer residual velocities, temporal aggregation, and multi-step
training. A deterministic neural residual alone is no longer a novel claim.

## Candidate

The source candidate is a **Bayesian reliability-gated residual twin**:

1. an automatically registered reusable physical twin supplies a forward prior;
2. an independently implemented O(3)-equivariant network predicts residual
   velocity and heteroscedastic variance;
3. an ensemble supplies epistemic variance;
4. residual-independent perception cues weight the likelihood, with correlated
   points grouped into effective observations;
5. action distance inflates variance outside source support;
6. a nested source-calibrated gate either admits the residual or returns the
   physical prediction exactly.

The model must learn from rollout error, but the innovation is used once in the
robust likelihood. It must not be recycled into prior perception reliability.

## Why this could exceed the current benchmark

Deform360 reports the following future-prediction reference values:

| Setting | Method | CD (m) | Track error (m) |
| --- | --- | ---: | ---: |
| Multi-episode | ParticleFormer | 0.051 | 0.079 |
| Multi-object | ParticleFormer | 0.038 | 0.048 |

PhysTwin is omitted from both settings because it lacks automatic episode and
object transfer. The proposed method attacks that omission directly. Unlike
PGRD's per-object training on about 100 overlapping episodes, the intended model
amortizes residual dynamics across objects and adapts an object-level posterior
from a small number of source interactions.

Accuracy alone is not sufficient. The intended differentiator is a calibrated
answer to *when the learned residual should be trusted*. The decisive comparison
is deterministic residual versus probabilistic residual versus gated residual,
with NLL, energy score, coverage, and fallback frequency reported alongside CD
and track error.

## Development boundary

The 27 outcomes from `002-rope-silk`, `083-blanket-cloth`,
`085-scarf-cloth`, `092-squirrel`, and `170-spider` are already open and may be
used only for development. Evaluation is leave-one-object-out, with trust
thresholds selected inside each outer training fold.

Penguin episodes 0, 2, 5, and 8 remain sealed. PokeFlex targets remain sealed.
The frozen Causal4D claim and artifacts remain unchanged.

## Proceed gate

A new independent preregistration is justified only if the cross-fitted gated
arm simultaneously achieves:

- at least 5% future track and Chamfer improvement over the strongest admissible
  fallback;
- at least 5% late-horizon improvement in both metrics;
- no episode degraded by more than 10%;
- 90% coverage between 85% and 95% at the episode-clustered level;
- improved energy score;
- exact physical output whenever the residual is rejected.

Failure leaves persistence or the physical twin as the honest fallback and does
not unlock any held episode.

## External scaling track

The public Deform360 host currently exposes approximately 190 object directories,
far beyond the ten staged locally. A stratified 1D/2D/3D expansion is the route
to a genuinely universal model. It must be processed under a separate data and
split lock; the 27 development episodes cannot also serve as a final benchmark.

The exact official Deform360 train/test episode and object lists are not released
with the baseline code. A direct Table 4 or Table 5 claim therefore requires the
authors' split or an explicitly labeled independent protocol.

## Intervention geometry correction

The first executable residual probe exposed a concrete implementation fault:
conditioning on the end-effector origin placed the apparent controller roughly
16--23 cm from several tracked objects. The opt-in source runner now consumes
Deform360's official UMI gripper-taxel geometry from its MIT-licensed checkout,
keeps a deterministic set of surface identities through time, and computes
contact proximity in metric space. The legacy wrist-origin loader remains the
default compatibility path.

Taxel density is not evidence. Contact activation uses the maximum local
proximity while controller offsets and velocities use a normalized weighted
average, so duplicating an identical surface block cannot increase the modeled
contact strength. Tactile outcomes remain excluded from predictive inputs.

The executable source arm also consumes the prediction-first, checksummed
`prediction.npz` produced by the already-frozen graph-action-support pipeline.
Its readout has the same material-point identities as `pcd_clean`, allowing the
residual to learn on top of actual driven-minus-zero Warp dynamics rather than
the initial inertial smoke placeholder. Inner source calibration chooses
between persistence and this physical prior before the outer object is scored;
the held outcome never selects its own fallback.

Residual admission is calibrated against the physical prior because rejection
returns that prior exactly. For the raw-Warp compatibility arm, inner source
calibration may still choose persistence. For the trusted-Warp arm, the frozen
episode-level trust policy has already made that decision: a zero scale is
byte-identical persistence, while a positive scale admits a conservative Warp
response. A second global object-level decision must not erase those
cross-fitted episode decisions. The residual therefore falls back to the
trusted trajectory, which itself may be persistence.

## Trusted-Warp composition result

The five leave-one-object-out folds were rerun for 2,000 optimization steps on
a deterministic 256-node subset. The aggregate result over 27 episodes is:

| Arm | Future track (m) | Future CD (m) | Late track (m) | Late CD (m) |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 0.011984 | 0.007803 | 0.017168 | 0.010454 |
| Trusted Warp | **0.011190** | **0.007393** | **0.015538** | **0.009556** |
| Deterministic residual | 0.011986 | 0.007826 | 0.017132 | 0.010291 |
| Gated residual | **0.011190** | **0.007393** | **0.015538** | **0.009556** |

Trusted Warp improves the four metrics over persistence by 6.62%, 5.26%,
9.49%, and 8.59%, respectively, with 3.33% maximum episode-level future
degradation. Every residual fold selected `utility_threshold=1.1`, the exact
abstention arm. The current residual therefore adds no transferable value and
should not be scaled.

This is still source development, not a state-of-the-art claim. It uses a
shorter horizon and a point subset, and it does not evaluate calibrated
coverage. The justified scaling candidate is the observable trust-gated
physical model, not this residual architecture.
