# Sparse Observation Budget on an Already-Open DEFORM Case

## Question and Scope

Can choosing a small number of informative prefix measurements improve a
successful frozen predictor's hidden future more than equally costly random,
spatial, variance-based, or global-information selection?

This is one exploratory development experiment, not another confirmatory
protocol. It uses the lexicographically first trajectory (`103.pkl`) in the
already-open, 14-case DLO2 v7 DEFORM/local-residual prediction archive. The
archive SHA-256 is fixed in
`configs/sota/deform_sparse_observation_budget_dev_v1.json`. Neither the DEFORM
backbone nor its successful source-trained local-residual correction changes.
No DLO4/DLO5, held-v8, fresh target, camera provider, or robot execution is used.

## Matched Measurement Budget

All indices below are original zero-based dataset frames and identities.
The archived predictions start at frame 2, after two initializing states.
Known clamped-node action trajectories remain allowed by the original DEFORM
contract; this experiment is not an unknown-action forecast.

- Prefix measurement times: 9, 17, 25, 33, 41, 49.
- Eligible identities: 0, 2, 4, 6, 8, 11; 0 and 11 are controlled endpoints.
- Scored identities: 3, 5, 7, 9, disjoint from every eligible measurement.
- Forecast: frames 50 through 169 inclusive, equally weighted.
- Budgets: 0, 1, 2, 4, 8 three-dimensional point-frame observations.
- Random selection: 32 seeded nested orders. Other policies: one deterministic
  nested order. No policy uses measurement values to choose its schedule.
- Score all planned budgets; do not select a favorable budget after scoring.

The five policies are random, spatial/temporal farthest sampling, maximum
remaining observation variance, nuisance-marginalized global information gain,
and nuisance-marginalized future-query variance reduction. All receive the same
candidate set, cost, prior, noise model and update. Controlled endpoint samples
can help identify a shared measurement bias; they do not update the free-node
field directly. Random and spatial sampling can spend budget on those endpoints
even in the native condition. Comparisons with the information and variance
policies therefore matter more than beating random alone.

## Diagnostic Belief Model

The archived predictor supplies a mean and source-fitted coordinate marginal
variance. A rank-4 Dirichlet chain basis, weighted by inverse mode index and
normalized per free-node row, supplies an explicitly assumed cross-node
correlation. Its amplitude at each point, coordinate and time is the square
root of that archived marginal variance. The 12 standardized latent
coefficients have an identity Gaussian prior and are held fixed in time.
This is a readout residual, not a simulator state correction or learned
material dynamics. The time dependence comes only from the frozen marginal
scales, not a fitted transition operator.

The future query is the mean coordinate variance on the four hidden nodes over
the entire declared horizon. QR compression preserves its exact trace objective
without constructing a full trajectory covariance. Each selected three-vector
conditions the same Gaussian model. A fixed 1 mm independent observation scale
regularizes the update. There is no measurement clipping or confidence inferred
from the size of a state innovation.

Two conditions are declared before the run:

1. **Native annotations:** use the released prefix coordinates directly, with
   no injected noise. The 1 mm scale is a modeling assumption, not an estimate
   of annotation accuracy. Correlations or bias in those annotations are not
   validated by this experiment.
2. **Simulated shared bias:** add a single 3D Gaussian translation with 5 mm
   coordinate standard deviation to all candidate measurements, plus 1 mm
   independent point noise. Use 16 fixed draws shared across all policies.
   Include the shared translation as a nuisance variable and marginalize it;
   do not add it to the physical readout. This is a matched-model synthetic
   stress test, not evidence of removing actual camera bias.

The zero-budget mean retains the exact original dtype, shape, and C-order
bytes. Conditional means do not feed back into or overwrite the DEFORM replay.
No accuracy or calibration monotonicity is assumed for actual errors.

## Execution and Outputs

Commit the new source, tests and config before the real case is scored. Run:

```bash
PYTHONPATH=src python scripts/run_deform_sparse_observation_budget.py \
  --archive /path/to/hash-verified/official_prediction.npz \
  --config configs/sota/deform_sparse_observation_budget_dev_v1.json \
  --output /path/to/new/development-run
```

The runner refuses a dirty source worktree, incorrect archive digest, changed
case-selection rule, overlapping observed/scored identities, unmatched budget,
or existing output root. It records source hashes and the clean Git revision.
It writes selection schedules before reading permitted prefix measurements,
then seals all predictions before handing the hidden future to scoring.

Primary error is mean absolute coordinate error in mm on the hidden future,
matching the coordinate-L1 convention of the DEFORM experiment. Secondary
outputs are 3D point RMSE and early/middle/late coordinate L1. Marginal Gaussian
coverage, full interval width, coordinate NEES and per-point diagonal Gaussian
NLL use an additional fixed 1 mm score-noise floor and are descriptive only.
They are not a cross-identity joint distribution or a calibration claim.

Outputs include `results.json`, `curves.csv`, PNG/PDF error-versus-budget plots,
a short report, immutable input/selection/prediction records, and file hashes.
Random orders and synthetic bias draws are not independent physical trials.
No significance test or generalization interval is reported for this one case.

## Interpretation

The useful positive result would be lower hidden-future error at the same
budget than the strongest simple selector, not merely improvement over the
zero-measurement baseline. If global information or maximum variance ties or
beats the query-aware policy, report that directly: this experiment would not
establish a special advantage of query-aware selection. If sparse updates
degrade the good baseline, preserve that negative result without retuning the
basis, noise scale, horizon, nodes, case, or budget on its future outcomes.

This experiment can motivate a narrowly specified follow-up on already-open
development trajectories. It cannot by itself justify an ICRA acceptance
prediction, point-SOTA claim, fresh-object transfer, or calibrated robot safety.
