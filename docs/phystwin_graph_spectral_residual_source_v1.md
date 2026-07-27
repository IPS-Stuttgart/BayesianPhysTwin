# Graph-spectral discrepancy source gate

## Purpose

This experiment tests the strongest low-capacity successor left after the
spring-family, graph-local nonlinear, structural, constant-force, and
camera-only discrepancy failures. It asks whether discrepancy velocity has a
shared action-conditioned structure across interactions, while preserving
endpoint persistence as an exact fallback.

This is an exploratory model-family gate on 17 already-open source
interactions. It is not an independent result and it does not authorize access
to any listed target artifact.

## Model

Each interaction uses a deterministic prefix-supported farthest-point sample
of at most 128 material identities. A symmetric normalized kNN Laplacian
provides graph modes. Mode zero is kept separate and the remaining modes are
divided into three frequency groups.

Within frequency group \(g\), the residual velocity follows

\[
\dot c_{t,g}
=
\rho_g \dot c_{t-1,g}
+ \beta_g a_{t,g}
+ \gamma_g(a_{t,g}-a_{t-1,g}).
\]

The coefficients are scalar and shared across xyz. The model is therefore
equivariant to global coordinate rotations, invariant to graph-eigenvector
sign choices, and low capacity: twelve transition coefficients for four
frequency groups.

A source prior is fit from complete outcomes outside the held-out fold. The
held-out prefix performs a ridge-MAP update toward that prior. No held-out
future object observation enters the update. Known controller actions may
extend through the forecast interval.

The model forecasts only the change from the dense prefix endpoint. That
change is inverse-distance lifted from spectral anchors to all material
identities and blended with the unchanged dense endpoint field. Blend zero is
exact endpoint persistence before the existing 10 mm cap and official
evaluation.

## Selection and gate

The three whole-case folds match the prior nonlinear source protocol. The
frozen candidates vary graph rank, prefix smoothing, prior strength, and
dynamic blend. Selection uses only cross-fitted source validation suffixes.

Advancement requires all of:

- at least 3% case-balanced mean improvement over endpoint persistence;
- lower aggregate Chamfer distance and manual-track error;
- joint improvement of both metrics in every case of at least two folds;
- no single case/metric ratio above 1.05.

Failure leaves endpoint persistence unchanged and keeps all target cases
closed. Passage would justify a new, separately hashed evaluation protocol,
not direct target access under this source protocol.

## Calibration and causal boundary

This gate evaluates conditional means only; it does not establish calibrated
spectral-process uncertainty. Complete outcomes are allowed only for
source-prior training cases. A scored held-out case supplies residual evidence
through its prefix and controller actions through its validation suffix.
Future held-out object observations, Prob4D trajectories, and target artifacts
are forbidden.

The experiment is separate from frozen Causal4D claims and from held-v8. It
must not inspect or modify any held-v8 target, query, score, barrier, outcome,
or process artifact.
