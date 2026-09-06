# DP residual experts: completed DLO1 development pilot

**Decision: do not replace the current correction or promote a DP accuracy claim.**

A truncated Dirichlet-process mixture gate was implemented and executed against
matched finite-mixture, finite-Bayesian-mixture and single-expert controls.
It improves over the frozen DEFORM hybrid, but **does not improve the existing
BayesianPhysTwin local residual correction**.

## Recorded outcome

Mean coordinate L1 over 498 forecast frames and eight historical DLO1 validation
trajectories, in millimetres; lower is better:

| Method | Mean L1 (mm) |
|---|---:|
| Frozen DEFORM hybrid | 9.256205 |
| Existing BayesianPhysTwin local residual, ridge 1 / shrinkage 0.5 | **8.288815** |
| Newly inner-tuned single residual expert | 8.575966 |
| Inner-tuned finite mixture (selected K=2) | 8.624752 |
| Inner-tuned finite Bayesian mixture (selected K=4) | 8.567170 |
| Truncated DP mixture (maximum K=8) | 8.538863 |
| Action-aware geometric persistence | 51.961295 |

The DP improves on the frozen hybrid by 7.75%, with eight wins out of eight.
However, it is **3.02% worse than the existing local correction**, losing all
eight paired trajectories. The mean DP-minus-existing difference is +0.250048 mm;
the descriptive paired trajectory-bootstrap 95% interval is [+0.117985,
+0.392036] mm.

The smaller DP advantages over the newly tuned single expert, finite mixture,
and finite Bayesian mixture are 0.037103, 0.085889, and 0.028307 mm respectively.
All three descriptive paired intervals include zero. Beating the frozen hybrid
is therefore not evidence that the DP improves the developed system or earns
its additional complexity.

The full result also retains prespecified 50/150/498-frame prefix diagnostics.
The DP is better than the existing correction over the first 50 frames
(4.877318 versus 5.039958 mm), but worse over 150 and 498 frames. This secondary
horizon result does not override the full-horizon primary comparison.

## What was actually tested

This is a **first-stage kinematic mixture-of-residual-experts pilot**, not a full
sticky-HDP switching dynamical model. One gate shared by all material nodes uses
only permissible kinematic features: initial state, supplied clamp motion and
the frozen hybrid rollout. A scaler and up-to-eight-dimensional PCA are fitted
on training features, followed by a Gaussian-mixture gate. The DP arm uses
scikit-learn's truncated variational stick-breaking prior. Its node-specific
ridge experts are partially pooled toward the common residual model.

The finite and finite-Bayesian controls use the same features, PCA, covariance
family, expert parameterization, shrinkage bank and seed-averaging policy.
The Bayesian finite control isolates changes beyond merely using a Bayesian
mixture estimator. The DP's selected concentration is 0.1, ridge is 100, and
shrinkage is 0.25. Its final fits retain seven/eight/eight components above
weight 0.01 across the three seeds, often reaching the truncation cap. These
counts are not identified physical modes.

No transition model, residual-history-conditioned latent-state inference,
HDP, calibrated uncertainty or physical mechanism recovery was evaluated.
This result must not be generalized to those untested models.

## Data and comparison boundary

Only the original **40 DLO1 fit and eight historical validation trajectories**
were used. Thirty-two fit trajectories train candidate models and eight
hash-assigned fit trajectories choose hyperparameters. Selected models are
refitted on all 40 before scoring the original eight validation trajectories.
All 48 causal queries were distinct, and no exact causal-query duplicate crossed
the fit/validation boundary. Three prespecified mixture seeds (7, 23, 61) are
averaged; a favorable seed is not selected from validation outcomes.

The baseline is the frozen **retrained DEFORM hybrid including DEFORM's own GCN
correction**, not bare rod physics. Every arm receives the same two observed
initial states, known future clamped-node trajectory and hybrid rollout. Future
free-node truth is not a query feature. At this initial-state cutoff the
permitted last residual is zero, so `last_residual_initial_zero` is identical
to the frozen hybrid. This is not a nonzero-prefix last-residual comparison.

The existing ridge-1/shrinkage-0.5 correction had historically been selected
using this validation set. In contrast, the new models' selection uses only
the inner split of the original fit set. Accordingly, the retained comparison
answers whether this newly selected procedure improves the existing method on
these recordings; it is not an unbiased fresh comparison of every model family.

Held validation targets are present in the prepared NPZ, not in an unopened
remote lockbox. They are excluded from new-model fitting and hyperparameter
selection; forecasts are saved and hashed before the final held scoring step.
The record does not claim target bytes were unread before that step.

Source-test data, official evaluation partitions and other DLOs were not read.
The eight validation trajectories had been examined historically, and all
reported intervals are descriptive development diagnostics for a single DLO.
No fresh independent confirmation, benchmark promotion or paper claim changes.

## Execution and verification

- [Successful GPU preparation, run 34052781162](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34052781162).
- [Successful comparison and seven tests, run 34053092767](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34053092767).
- Exact comparison implementation: `7645f02265cc5087221f537742f06a36a8c245ba`.
- [Compact retained result and artifact hashes](result_summary.json).

Preparation reproduces the frozen validation baseline within its original
1e-7 m tolerance. The custom single expert matches the production local
predictor at ridge 1/shrinkage 0.5 to 1.48e-14 m. All seven focused tests pass,
including synthetic regime recovery as a prediction check, genuine DP versus
finite prior distinction, future-truth feature isolation, exact zero-strength
fallback and unchanged clamped nodes. All selected final gate fits converge.
Saved predictions were also independently rescored locally; every reported
per-trajectory point metric and the prediction hash verified.

The initial preparation attempt stopped at Git's ownership check before data
loading. A job-scoped trust entry for the exact frozen upstream checkout fixed
that infrastructure issue; it was not an outcome-driven scientific retry.

## Reproduce the comparison without GPU inference

Download preparation artifact `9995078685` from its successful workflow run
and extract `development.npz` and `preparation.json` into `data/`. Its ZIP SHA256
is `083eaca1233d6373e515790f567a831a1d9d6092069ff2cd244e10955ae45c60`.
At the exact implementation commit above, using Python 3.12:

```bash
python -m pip install -e . numpy==2.2.6 scipy==1.15.3 scikit-learn==1.8.0 pytest==8.3.5
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m pytest -q experiments/dp_residual_development_v1/test_model.py
python experiments/dp_residual_development_v1/run.py \
  --bundle data/development.npz \
  --preparation data/preparation.json \
  --output fresh-replay-output
```

The output directory must not already exist. Full comparison artifact
`9995164466` includes the selection bank, sealed forecasts, scores, log and
exact implementation source archive. Its ZIP SHA256 is
`3b7fdeb0400f200f1346f50c6d7573c27a197def62bb4d13d5eefa4d09c59440`.

This pilot is retained as a bounded negative finding for incremental DP value.
No hyperparameters were revised after observing its held-validation scores.
