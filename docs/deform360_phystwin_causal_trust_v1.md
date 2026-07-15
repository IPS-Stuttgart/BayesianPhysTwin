# Deform360 PhysTwin causal-trust diagnostic v1

**Status:** superseded by the contact-regime-gated v2 protocol after episode 3
falsified global transfer. This file preserves the rule and evidence order that
were in force before that result was observed.

## Scope

This is an exploratory, source-only method-development track for filling the
empty PhysTwin multi-episode cell in Deform360. It is separate from the frozen
Causal4D main-paper claim. Episodes 0, 2, and 8 remain calibration-only and
episode 5 remains sealed target data. No result in this note is a Deform360
state-of-the-art claim.

The locked source actions for `081-stripe-rope` are episodes 1, 3, 4, 6, 7,
and 9. Episodes 1 and 4 were examined while developing the method. The
remaining four actions provide the first wider transfer check, but the entire
six-action panel remains exploratory because the model family was chosen after
seeing two source actions.

## Motivation

The official PhysTwin rollout exhibits two effects that should not be assigned
one trust value:

1. the deformation caused by the commanded intervention; and
2. autonomous settling caused by gravity, support, and rest-state mismatch.

For each source action, run official Warp twice from the same registered frame
zero:

- `X_driven`: the measured Deform360 controller trajectory;
- `X_zero`: the controller held at its frame-zero position.

The prediction is

```text
X_hat = X_observed,0
      + a * (X_driven - X_zero)
      + b * (X_zero - X_zero,0)
```

where `a` is intervention-response trust and `b` is autonomous-drift trust.
The zero-action rollout is a causal control variate: it removes simulator drift
that is shared by the factual and no-action worlds without observing the
future object state.

The grid contains `a,b in {0.0, 0.1, ..., 1.0}`. It includes exact controls:

- `(0,0)`: persistence;
- `(1,1)`: raw PhysTwin;
- `(1,0)`: full intervention response with autonomous drift removed.

## Frozen source rule

The first 60 of 76 processed frames are source-training frames. The last 16
frames are untouched source tails.

Candidate selection minimizes the execution-balanced mean of:

```text
0.5 * (track_RMSE / persistence_track_RMSE
     + Chamfer / persistence_Chamfer)
```

on source-training frames only. Source tails never select weights. For the
six-action panel, leave one action out, select `(a,b)` using the other five
training intervals, and evaluate only the held action's untouched tail.

The wider source gate passes only if all of the following hold:

1. pooled leave-one-action-out tail track improvement is at least 5%;
2. pooled leave-one-action-out tail Chamfer improvement is at least 2%;
3. both metrics improve on at least four of six held actions;
4. neither metric degrades by more than 10% on any held action;
5. every hull, support transform, contact association, driven rollout, and
   zero-action rollout passes its registered QA checks;
6. mutating source-tail values cannot change any selected fold weight.

Failing this gate freezes the method as a negative source diagnostic. Passing
it permits uncertainty calibration on episodes 0, 2, and 8, followed by one
sealed episode-5 evaluation. It does not by itself permit a benchmark-wide
SOTA claim; that requires the official multi-object protocol.

## Two-action development result

The source-train-selected weights are:

```text
intervention-response trust a = 0.4
autonomous-drift trust b      = 0.1
```

On the untouched tails of episodes 1 and 4, pooled results are:

| Method | Track RMSE (mm) | Chamfer (mm) |
|---|---:|---:|
| Persistence | 12.966 | 9.537 |
| Raw PhysTwin | 14.957 | 13.121 |
| Causal trust `(0.4, 0.1)` | **11.271** | **8.947** |

This is a 13.07% track and 6.18% Chamfer improvement over persistence. The
result is encouraging but insufficient: it uses only two examined source
actions and therefore has no independent claim status.

## Rejected shortcuts

- A scalar output blend improves the two tails but is only a kinematic
  model-form diagnostic.
- Scaling controller displacement inside Warp improves track error but worsens
  Chamfer, so a single actuation gain does not explain the discrepancy.
- Separating virtual attachment stiffness from object stiffness does not beat
  persistence; the attachment is already in a near-saturated regime.
- Mass scaling and a fixed global parameter grid do not solve the source error.

These controls motivate retaining separate uncertainty over intervention
response and autonomous simulator drift rather than relabeling either as an
identified material or robot parameter.

## Claim boundary

The trust weights are source-calibrated model-form variables. They are not
material constants, grasp stiffnesses, measured actuator gains, or evidence
that PhysTwin alone beats Deform360 baselines. The target rollout may use the
known controller trajectory and target frame-zero registration, but it may not
use any future target object observation.
