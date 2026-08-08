# Deform360 calibration factor materializer

## Purpose

This stage converts existing public Deform360 prefix measurements into the
actual solver inputs required by the locked ten-object calibration observability
study. It requires no new capture and no human registration approval. Its inputs
come from the public dataset and frozen source-side processing:

- synchronized RGB prefixes exported through calibrated, claim-bearing Prob4D;
- synchronized unitless tactile response grids;
- the released robot end-effector poses and gripper openings;
- the exact UMI taxel geometry implemented by the pinned Deform360 processing
  revision;
- a row-bound physical prediction and PhysTwin state linearization; and
- a shared, predeclared physical-query Jacobian.

The materializer never reads confirmation objects, future visual frames, or
target outcomes.

## Contact reduction

Deform360 tactile values are peak-relative responses, not forces and not
Cartesian displacement measurements. For one active gripper/contact episode,
let (a_k \ge 0) be the positive response at taxel (k), and let
(p_k \in \mathbb R^3) be that taxel's world position from the synchronized
robot pose and exact gripper geometry. The reduced contact location is

\[
\bar p = \frac{\sum_k a_k p_k}{\sum_k a_k}.
\]

The metric innovation is formed once as

\[
r = \bar p - p_{\mathrm{physical}},
\]

where (p_{\mathrm{physical}}) and its state Jacobian come from the same frozen
physical-prefix replay. The response magnitude is used only to locate the
contact patch. It is not interpreted as force or converted into confidence.

The row covariance is the full weighted patch scatter plus a fixed 5 mm
localization floor. It is deliberately **not** divided by the number of active
taxels. Duplicating an identical taxel block therefore leaves the centroid,
covariance, reliability, and posterior information unchanged. At least two
geometrically distinct active taxels are required.

All rows from one sensor/contact episode share one correlation group. The
registered posterior configuration caps that group at one effective sample.
Each sensor also receives a shared three-dimensional bias with a 10 mm prior
standard deviation, so coherent robot/registration bias cannot masquerade as
independent state information.

Prior reliability is supplied from residual-independent source QA. Neither the
visual PhysTwin innovation nor the contact PhysTwin innovation enters prior
reliability or nominal contact probability.

## Posterior materialization

The posterior command accepts only a Prob4D provider-v2 artifact with:

- explicit claim-bearing attestation;
- calibrated point and gauge covariance identities;
- joint cross-window gauge covariance;
- exact runtime revision evidence; and
- a causal prefix whose declared frames are strictly before the cutoff.

Exploratory or uncalibrated Prob4D exports fail before the visual innovation is
formed. The same physical prior is then updated twice:

1. visual explicit-gauge factors only; and
2. the identical visual factors plus the grouped contact anchor and sensor-bias
   nuisance.

The existing robust gauge-aware likelihood processes each innovation once. Its
state covariance block is inverted to produce the nuisance-marginalized
reference and candidate state precisions. An inadmissible numerical update uses
the solver's exact prior fallback and records the reason. A materially negative
candidate information increment is retained as a completed non-evaluable
result; a zero increment is an evaluable no-gain result. Neither is projected
into a positive result.

## Commands

The first command consumes arrays derived from the public tactile and robot
prefix and writes a content-addressed non-pickled anchor:

```bash
python scripts/science/materialize_deform360_calibration_factors.py \
  contact-anchor \
  --object-id OBJECT \
  --observation-case-id CASE \
  --episode-id 0 \
  --causal-frame-stop 58 \
  --frame-ids contact/frame-ids.npy \
  --sensor-names contact/sensor-names.json \
  --contact-episode-ids contact/contact-episode-ids.json \
  --tactile-response contact/tactile-response.npy \
  --taxel-world-positions contact/taxel-world-positions-m.npy \
  --physical-patch-prediction contact/physical-patch-prediction-m.npy \
  --state-jacobian contact/contact-state-jacobian.npy \
  --source-reliability contact/source-reliability.npy \
  --source-revision DEFORM360_REVISION \
  --source-artifacts contact/source-artifacts.json \
  --output contact-anchor.npz
```

After source-only point/gauge calibration has produced a claim-bearing Prob4D
belief, the second command publishes the exact observability inputs:

```bash
python scripts/science/materialize_deform360_calibration_factors.py \
  posterior \
  --observation-belief calibrated-prob4d-belief.npz \
  --physical-linearization physical-linearization.npz \
  --physical-prediction physical-prediction-m.npy \
  --contact-anchor contact-anchor.npz \
  --physical-query-jacobian shared/physical-query-jacobian.npy \
  --state-prior-covariance state-prior-covariance-m2.npy \
  --output-dir materialized-object
```

The output directory contains:

```text
SHA256SUMS
candidate-marginal-precision.npy
contact-anchor.npz
materialization.json
physical-query-jacobian.npy
reference-marginal-precision.npy
```

These filenames map directly to an evaluated row in
`build_deform360_calibration_observability_batch.py`. Publication is
no-overwrite and atomic at the directory boundary.

## Failure and claim boundary

Missing contact, fewer than two distinct active taxels, case/cutoff mismatch,
uncalibrated Prob4D covariance, row-order mismatch, non-positive covariance, or
non-monotone candidate information cannot be repaired by replacing the object.
The object remains a retained technical or support-negative row in the locked
ten-object denominator.

This artifact establishes only calibration-source factor construction and
observability. It does not establish future prediction accuracy, tactile
benefit, confirmation, deployment safety, official Deform360 benchmark parity,
or state of the art.
