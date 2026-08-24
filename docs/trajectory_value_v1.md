# Whole-trajectory proper scoring and frozen decision value v1

## Purpose

Endpoint RMSE and marginal coverage can miss two properties that matter for a
physical twin:

1. whether the **joint future trajectory distribution** has the correct
   temporal and cross-coordinate dependence; and
2. whether the uncertainty changes a frozen action choice in a way that reduces
   held-out physical loss.

`trajectory_value_v1` supplies target-closed experimental utilities for these
questions. It lives in `bayesian_phystwin_experiments` and does not alter the
stable inference API, guard, fallback, or any registered cohort.

## Scaled trajectory score

A trajectory is vectorized after the caller freezes its time points,
coordinates, units, and coordinate scale. For predictive samples
$X^{(1)},\ldots,X^{(M)}$ and target $y$, the empirical energy score is

$$
\operatorname{ES}
=
\frac1M\sum_m \lVert X^{(m)}-y\rVert_2
-
\frac{1}{2M^2}\sum_{m,n}
\lVert X^{(m)}-X^{(n)}\rVert_2.
$$

The implementation computes the second term in deterministic blocks to avoid a
single $M\times M\times d$ allocation.

The optional registered variogram score is

$$
\operatorname{VS}
=
\sum_{(i,j,w)\in\mathcal P}
w\left(
|y_i-y_j|^p
-
\frac1M\sum_m |X_i^{(m)}-X_j^{(m)}|^p
\right)^2,
$$

where the finite pair roster $\mathcal P$, weights, power $0<p<2$, and
coordinate scale are all frozen before target access. Registering time-separated
and cross-coordinate pairs makes the score sensitive to coherent motion,
oscillation, drift, and temporal coupling that marginal intervals cannot test.

The reported total is

$$
S = \lambda_{\mathrm{ES}}\operatorname{ES}
  + \lambda_{\mathrm{VS}}\operatorname{VS}.
$$

For a claim-bearing study, the complete score definition—including the two
weights—must be source-frozen. Do not tune them after target scoring.

## Frozen action decision value

For a finite sorted action roster $\mathcal A$, a method supplies predictive
loss samples $L^{(m)}(a)$ before target access. The selected action is

$$
a^\star
=
\arg\min_{a\in\mathcal A}
\frac1M\sum_m L^{(m)}(a),
$$

with lexicographic tie-breaking inherited from the sorted action roster. After
one authorized outcome opening, the record binds the realized loss
$L_{\mathrm{real}}(a)$ for every registered action and reports

$$
\operatorname{Regret}
=
L_{\mathrm{real}}(a^\star)
-
\min_a L_{\mathrm{real}}(a).
$$

The loss definition must be simple and fixed before target access. An example is
endpoint error plus a preregistered penalty for excessive deformation or contact
loss. The utility accepts already computed loss samples so it does not need to
serialize an arbitrary Python objective.

## Example

```python
import numpy as np

from bayesian_phystwin_experiments.trajectory_value_v1 import (
    FrozenActionDecisionValueV1,
    TrajectoryProperScoreConfigV1,
    TrajectoryProperScoreV1,
    VariogramPairV1,
)

config = TrajectoryProperScoreConfigV1(
    score_definition_id=score_definition_id,
    coordinate_scale_id=coordinate_scale_id,
    coordinate_scale=np.full(6, 0.01),
    energy_weight=1.0,
    variogram_weight=0.25,
    variogram_power=0.5,
    variogram_pairs=(
        VariogramPairV1(0, 3, 1.0),
        VariogramPairV1(1, 4, 1.0),
        VariogramPairV1(2, 5, 1.0),
    ),
)

score = TrajectoryProperScoreV1(
    config=config,
    prediction_artifact_id=prediction_artifact_id,
    target_artifact_id=target_artifact_id,
    object_session_id="sloth/session-03",
    action_id="pull-fast",
    arm_id="guarded_physical",
    predictive_samples=trajectory_samples,
    target_trajectory=target_trajectory,
    prediction_sealed_before_target=True,
)

decision = FrozenActionDecisionValueV1(
    decision_protocol_id=decision_protocol_id,
    loss_definition_id=loss_definition_id,
    prediction_batch_id=prediction_batch_id,
    target_access_attestation_id=target_access_attestation_id,
    object_session_id="sloth/session-03",
    method_id="guarded_physical",
    action_ids=("hold", "pull-fast", "pull-slow"),
    predictive_loss_samples=predictive_loss_samples,
    realized_losses=realized_losses,
    predictions_sealed_before_target=True,
)
```

## Recommended registered comparison

On identical complete physical object/session units, report:

- unchanged physical fallback;
- `last_residual`;
- one frozen discrepancy-only candidate;
- one frozen guarded physical candidate;
- trajectory energy and variogram scores;
- selected action, realized loss, and regret;
- exact-fallback count and reasons; and
- worst-session regret and the object/session-clustered paired interval.

Frames, points, time steps, views, and action alternatives are nested
observations. The independent statistical units remain complete physical
object/session groups.

The existing `CrossActionTransportResultV1` can use the combined trajectory score
as its lower-is-better registered score. Decision-regret records should be
analysed separately so a downstream action result cannot rescue failed provider,
identifiability, calibration, or physical-transport gates.

## Scientific boundary

These utilities establish only the computed score and finite-action regret under
the exact supplied samples, target, scales, pair roster, loss definition, and
access attestations. They do not establish:

- calibrated deployment uncertainty;
- a universally proper arbitrary weighted composite;
- safe or optimal action execution outside the finite roster;
- causal identification or a unique physical cause;
- unseen-object transfer;
- real provider competence;
- completed Causal4D intervention evidence;
- deployment safety; or
- state of the art.
