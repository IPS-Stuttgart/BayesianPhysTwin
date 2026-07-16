# Deform360 reusable dynamics protocol 081 v1

## Scientific question

The first state-of-the-art target is the empty PhysTwin cell in Deform360's
multi-episode setting. This protocol asks the prerequisite question on one rope:

> Does one physical parameter tuple selected across several source actions
> transfer better to unseen executions than tuples selected from one source
> action at a time?

This is a same-object reusable-twin competence test. It is not yet an official
multi-object benchmark result and cannot support a state-of-the-art claim by
itself.

## Prospective split

The `081-stripe-rope` episodes have fixed roles:

- source selection: 1, 4, and 6;
- independent calibration: 0, 2, and 8;
- sealed target: 5.

The protocol was checksummed before reading calibration frame 110 or any
calibration dynamics outcome. Episode 5 remains inaccessible through both the
source and calibration request validators.

The dynamics slice is raw frames `[110, 191)`. The external tracking pipeline
produces 76 usable frames after its five-frame tail rule. Frame 0 initializes the
twin and frames `[1, 76)` form the independent predictive horizon, split into
three equal 25-frame horizon bands.

## Association boundary

The previously frozen reusable-association v2 gate passed independently on all
three calibration episodes. Dynamics initialization therefore reuses its fixed
appearance-first, calibrated multiview policy, but runs it at raw frame 110.
Only frame 110 may be read for this operation. The accepted masks are sealed
before the full 81-frame slice is staged.

Association uses source appearance and calibrated geometry. It cannot use a
simulator residual, future frame, future metric, or target media.

The public aligned release does not contain the per-camera
`rendered_urdf.h5` products used to remove gripper pixels from depth. The
frozen public-data pipeline therefore retains any gripper depth that remains
inside the propagated object mask. This limitation is recorded for every
calibration execution and is not repaired or tuned after prediction outcomes
are observed.

## Physical pooling and controls

The official PhysTwin revision and configuration are pinned in the protocol.
The physical grid contains 24 tuples:

- spring stiffness: 10k, 30k, 50k, or 80k;
- drag damping: 1, 3, or 10;
- dashpot damping: 50 or 100.

Every candidate is scored on source frames `[1, 60)` using the
execution-balanced mean of normalized track RMSE and symmetric Chamfer distance.
One tuple is selected jointly over all three source executions. The matched
control selects a tuple separately from each source execution and applies all
three frozen choices to every calibration execution; their median is reported.
Source tails are diagnostic only and cannot alter either selection.

A candidate that becomes non-finite on any source execution remains in the
attempted-grid audit but is jointly ineligible. It is never replaced, repaired,
or tuned around after the failure is observed.

The cardinality-normalized action-trust rule is already source-frozen:

```text
x_hat = x0
      + (0.4 / controller_count) * (x_driven - x_zero)
      + 0.1 * (x_zero - x0)
```

Its weights cannot change after this lock. Raw Warp, persistence, pooled trust,
and matched single-source trust are all reported so physical pooling is not
confounded with the action-trust correction.

Before calibration scoring, the pooled tuple must also pass a source
compatibility check under that fixed trust rule: positive execution-balanced
tail transfer in both metrics, joint wins in at least two source executions,
and no source execution worse by more than 25% on either metric. This is a
sanity gate, not independent evidence, because the trust rule was developed on
the source cohort.

## Transfer gates

All three calibration executions are scored once after the source-selection
artifact is frozen. The primary pooled method must satisfy every registered
gate, including:

- at least 5% execution-balanced improvement in both CD and track error;
- joint CD/track wins in at least two of three executions;
- no execution worse by more than 10% on either metric;
- at least 3% late-horizon improvement in both metrics;
- match or beat the median single-source control in at least two executions;
- finite and physically plausible Warp trajectories.

The implementation also reruns the pooled driven and zero-action trajectories
once per calibration execution. The maximum coordinate RMSE between repeats
must remain below the registered 0.1 mm replay threshold. Object-spring strain,
direct controller-spring support, and frame-zero association distance are
reported as physical diagnostics rather than inferred from prediction error.

With only three calibration executions, the uncertainty check is deliberately
modest. A 75% execution-level split-conformal radius is rank 3 of 3, the maximum
calibration score. It is a target-opening safety gate, not evidence of 90%
coverage. Proper calibration claims require a larger independent panel.

If any conjunctive gate fails, the result is frozen and episode 5 stays sealed.
If all pass, the method and conformal radius are frozen before episode 5 opens.

## Frozen calibration outcome

The one-shot evaluation passed every accuracy, horizon, replay, strain, and
conformal-radius gate. Relative to persistence, the primary pooled method
improved execution-balanced full-horizon track RMSE by 22.12% and symmetric CD
by 17.13%; late-horizon gains were 25.40% and 20.87%, respectively.

It failed the pooling control: the pooled tuple matched or beat the median
single-source-selected control on both metrics in only one of three executions,
instead of the required two. The miss was sub-millimetric on episodes 0 and 2,
but the preregistered threshold is unchanged. Episode 5 therefore remains
sealed, and this experiment does not establish that pooled physical-parameter
selection is better than selecting from one source action.

The raw pooled Warp arm improved CD by 8.24% but worsened track RMSE by 0.24%.
The primary gain consequently belongs to the fixed trust/control-variate layer,
not to raw simulation alone. This distinction is carried into subsequent
Bayesian model-averaging work.

## Route to a state-of-the-art claim

Passing this protocol justifies a preregistered topology-stratified Deform360
evaluation with the full dense PhysTwin backend. The official comparison target
is ParticleFormer's multi-episode result (51 mm CD and 79 mm track error). A
credible SOTA result should beat that protocol-matched baseline while retaining
the pooling control, horizon analysis, replay-variance audit, and uncertainty
reporting defined here.
