# Forecast-aware Sparse Sensing: Opened-object Development Protocol

## Purpose and scope

Test two unproven extensions of the positive native-state/readout coupling:

1. Does physical propagation outperform equally informed temporal readout models?
2. Does a physics-conditioned observation schedule improve hidden future motion
   with the same eight point observations?

This is a new, separately frozen **development** comparison. It uses exactly the
parent's already-open DLO1/DLO2/DLO3 roster, checkpoints, readouts, material
identities, time alignment, and 30-prediction/29-analysis denominator. DLO2's
original design case remains excluded from analysis. The 16 DLO1/DLO3
trajectories are transfer of an update rule, not fresh confirmation or zero-shot
transfer of physical parameters. No DLO4/DLO5, official DLO3 evaluation, held-v8,
new Deform360 target, robot execution, or new recording is authorized.

The parent result and all original DEFORM source files remain byte unchanged.
There is no retraining, new checkpoint/seed selection, or automatic promotion.

## Information contract

Archive frame zero is raw frame two. As in the parent, raw frames zero and one
initialize the native physical state, and the prescribed four clamped-node
trajectories are known inputs. Future **free-node** values never enter a
prediction or query planner. The released containers include later frames; the
runner may decode those containers, but the method receives only the bounded
initial state, clamped inputs, and selected prefix queries.

The allowed measurement pool is the Cartesian product of archive times
25/33/41/49 and identities 2/4/6/8. Identities 3/5/7/9 are never queried after the
common two-frame initialization and are the only primary scoring identities.
One 3D material point at one time costs
one observation. Query identities are assumed known from the released motion
capture data. This is not evidence for automatic camera association or tracking.

Every schedule is computed from nominal simulated responses **before any
measurement value is revealed**, then executed in chronological order through a
write-once query interface. This first arm is model-conditioned preplanning,
not a claim of measurement-adaptive sensing. All budgets and policies use the
same available future clamped inputs. Random query plans have four fixed seeds;
they are scored individually, never averaged into a better point predictor.

## Reduced physical belief

Retain the complete native state at frame 25, including velocity, previous
positions, material frame and twist. Piecewise-linear material interpolation
defines twelve position and twelve velocity directions, zero at all clamps.
Their independent prior standard deviations are 10 mm and 100 mm/s. Symmetric
1 mm / 10 mm/s native perturbations provide a local response Jacobian through
frame 169; this is an empirical linearization, not a material model claim.

The query likelihood is relative to the frozen incumbent readout:

```text
observed point - incumbent point = native response * coefficients
                                 + shared translation bias + noise
```

The three shared-bias coefficients have 5 mm prior standard deviation; independent
measurement noise has 1 mm standard deviation. All policies use the same joint
linear Gaussian posterior. The nuisance bias is not applied to the physical
future. It remains a possible explanation of a common observation offset.
These fixed scales are a planning/inference model, not a source-calibrated or
validated predictive distribution. No NLL, coverage, or safety guarantee is
claimed by this experiment.

The posterior mean is mapped to native pose/velocity increments. A single radial
gain bounds the maximum point displacement to 30 mm and velocity increment to
300 mm/s, without altering clamps or clipping individual observations. Native
dynamics then propagate that state from frame 25. The final forecast is

```text
incumbent + updated native continuation - nominal native continuation
```

Thus the learned readout remains intact. Zero update returns the exact incumbent
object and bytes. This is prefix smoothing followed by prediction, not a future
observation update.

## Frozen comparisons

- Primary: eight queries greedily minimizing the model's expected average hidden
  future squared-position uncertainty over frames 50-169.
- Matched uniform: all four allowed identities at frames 41 and 49.
- Current-shape control: same greedy calculation for prefix-end free-node shape.
- Random: four fixed eight-query schedules.
- Secondary budgets: 4, 12 and 16 for uniform and forecast-oriented planning.
- Previous paired eight-observation pose/velocity update, reproduced exactly.
- Temporal controls using the same original eight observations: static residual,
  constant residual velocity, damped residual velocity, and decaying pose plus
  velocity. The latter two each use fixed time constants 0.1/0.3/1.0 seconds.
  Every control is reported; no hindsight choice of a time constant is promoted.

The planning objective projects posterior uncertainty onto hidden future
positions, including the effects of nuisance marginalization. It does not use
hidden measurement values. Query designs and their source-array hashes are
sealed before revealing observations. The budget-16 plans must be identical.

Clean predictions are primary. Secondary stress tests use eight predetermined
1 mm independent-noise repetitions, with and without one 5 mm translation shared
by all measurements in a trajectory. Every policy sees the same noisy candidate
bank. These are simulated sensor errors, not measured camera noise. Repetitions
are averaged within trajectory before aggregation and resampling.

## Decision and evidence order

The primary arm must, on **each** DLO1/DLO3 object:

1. Improve coordinate L1 and Euclidean point RMSE over the incumbent and uniform
   sensing, with at least 2% RMSE improvement over uniform sensing.
2. Beat every frozen temporal control on both point metrics.
3. Avoid increasing late-horizon RMSE relative to the incumbent.
4. Obtain at least five of eight joint trajectory wins over the incumbent.

These are development advancement criteria, not hypothesis-test significance
thresholds. The secondary budgets and noise conditions cannot rescue failure.
Independent evaluation needs a separate prospective authorization even on PASS.
No result from this study changes a previous decision or automatically opens a
protected dataset.

Use the parent's metrics, equal-trajectory/object summaries, horizon thirds,
10,000 whole-trajectory bootstrap replicates and seed. Comparisons are exploratory
and conditional on the opened objects. They are not official DEFORM or PhysTwin
leaderboard scores.

Before outcomes: commit code/protocol/tests, bind a clean source receipt, verify
all source archives, check native/no-op parity, generate all native response
models, seal all query plans, then reveal only each plan's queries and seal all
clean/noisy forecasts for every object. Only a complete prediction barrier opens
scoring. Retain any technical failure without dropping or replacing a case.
An independent analysis implementation must recompute the final metrics,
posterior coefficients, query plans, controls, hashes and decision.

## Novelty boundary

DEFORM already performs sensor-assisted estimation and studies sensor frequency.
Goal-oriented design and bias-aware filtering are established methods. The
proposed contribution is not those ingredients alone: this experiment asks
whether their integration with native dynamics and preserved learned readout
provides transferable, budget-matched predictive value.

- DEFORM: https://arxiv.org/abs/2406.05931
- Goal-oriented design: https://arxiv.org/abs/2102.06627
- Bias-aware filtering: https://arxiv.org/abs/2112.14432
