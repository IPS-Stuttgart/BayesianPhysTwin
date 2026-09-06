# DLO1 GP residual development diagnostic

Owner: issue #946. Classification: diagnostic, not a stable API or new claim.
This is a user-requested direct test of nonlinear and spatial residual learning.
It does not modify any frozen method or failed confirmation protocol.

## Data and comparison

Use the exact historical DLO1/train 40-fit / 8-validation / 8-source-test split,
the frozen 6400-update DEFORM hybrid checkpoint, all 498 recursive forecast
steps, and the existing canonical local features. This is post-open historical
method development, not fresh independent confirmation. No DLO2 (including its
sealed final group), official evaluation, or PokeFlex data is authorized.

Compare baseline, fixed ridge r1/s0.5, validation-tuned original ridge,
independent-node GP, material-coordinate coupled GP, and fixed-ridge plus each
GP. Use a Matérn-3/2 state kernel with training-median distance scale. The spatial
kernel multiplies this by a material-coordinate RBF of length 0.35. The bounded
pilot uses explicit Nyström approximations (80 anchors per independent node;
240 joint anchors). All training rows are used, with equal total likelihood
weight per trajectory/node. Landmark selection and feature/output scaling use
training data only. Exact causal query duplicates are collapsed as in the parent.
The six GP outputs jointly implement three raw residual coordinates and three
remaining-residual coordinates after the fixed ridge mean; output channels have
separate coefficients and are not a learned full vector-output covariance.

The approximation nugget is fixed at 1e-6, output scale floor at 1 mm. No adaptive
jitter, pseudoinverse or numerical repair is authorized. Working GP posterior
covariance is retained but **not assumed calibrated**. Mean predictions also
have a kernel-ridge interpretation and do not establish unique Bayesian value.

Every family's hyperparameters and the overall GP champion are selected on the
eight validation trajectories; selection.json is written before source-test
payloads are opened. Retain every selected-family source outcome, not only the
best source result. The diagnostic continuation rule uses the validation-chosen
champion: at least 1% mean improvement against tuned ridge, at least 5/8 paired
wins, and no case ratio above 1.1. It never authorizes a paper claim.

## Execution and provenance

The maintained residual-comparison workflow runs tests and this bounded pilot
on gpuserver4090 with existing environment/data/checkpoint, read-only GitHub
permissions, immutable action SHAs, and exact commit checkout. A raw-data audit
allowlist protects the DLO1 partition and delays historical source access.
Preflight, full validation bank, source predictions, per-trajectory and horizon
scores, input/output digests and implementation SHA are retained in the run.
Results, including negatives and failures, belong in the paper repository.

This pilot intentionally does not include a neural residual baseline, learning
curves, physical-state correction, or calibrated joint-uncertainty evaluation.
Those must not be claimed from this accuracy diagnostic.

Method references: Rasmussen & Williams, Gaussian Processes for Machine Learning
(2006), chapters 2 and 8; Borovitskiy et al., Matérn Gaussian Processes on Graphs,
AISTATS 2021 (motivation for topology-aware covariance; this rod pilot uses a
material-coordinate kernel, not the full graph-SPDE construction).
