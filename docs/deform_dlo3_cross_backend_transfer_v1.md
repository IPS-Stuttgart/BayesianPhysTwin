# DLO3 no-refit cross-backend coefficient transfer

## Scientific question

The completed DLO3 studies establish **procedure portability**: the same local
residual design and frozen hyperparameters improve both the official DEFORM
physical backend and PyElastica. They do not yet establish that the fitted
discrepancy itself is backend-independent, because each backend received its own
source-fitted residual coefficients.

This experiment asks a sharper question:

> Do local-residual coefficients fitted against DEFORM improve sealed
> PyElastica rollouts without any coefficient refit?

A positive result would indicate that the residual captures a component of the
missing dynamics shared by two physical backends, rather than only the error
signature of one simulator.

## Frozen arms

The registered source panel is the existing eight-trajectory DLO3
`train/source_test` partition. The official DLO3 evaluation remains unopened by
this experiment.

The comparison contains:

1. raw, source-frozen PyElastica predictions;
2. the already retained PyElastica-specific source-fitted correction;
3. unchanged DEFORM-fitted local residual models from seeds 42, 43, and 44,
   applied directly to the raw PyElastica predictions; and
4. the arithmetic prediction mean of the three no-refit transfer arms.

All residual models use the existing action-local feature representation, ridge
`1.0`, and shrinkage `0.25`. No DEFORM or PyElastica parameter, coefficient,
feature scaler, seed weight, trajectory, or threshold may be selected from the
cross-backend outcomes.

## Information order

The source-test outcomes were already opened by the predecessor DLO3 studies, so
this is a retrospective source-only diagnostic rather than fresh confirmation.
The implementation nevertheless preserves a closed execution order:

1. verify every source manifest, sealed PyElastica prediction, and DEFORM model
   by exact path, size, and SHA-256;
2. freeze the ordered source names, equal-seed aggregation, shrinkage, metric,
   bootstrap, and promotion gate;
3. write a method seal;
4. only then deserialize the eight source trajectory payloads and compute the
   no-refit predictions; and
5. report every seed, the equal-seed arm, and the backend-specific reference.

DLO3 official evaluation, DLO4, DLO5, and held-v8 are forbidden.

## Registered decision

The equal-seed no-refit arm is supported only when it:

- improves mean coordinate L1 over raw PyElastica by at least 1%;
- wins at least 6 of 8 complete trajectories;
- has no trajectory ratio above 1.10; and
- is accompanied by positive mean improvement for at least two of the three
  independently fitted DEFORM source models.

The evaluator additionally reports the fraction of the PyElastica-specific
correction gain retained without coefficient refitting. This is descriptive and
does not replace the conjunctive gate.

## Execution priority

The frozen DLO4/DLO5 procedure-level replication in Actions run `33361441865`
has exclusive priority on `gpuserver4090`. This branch deliberately adds no
request file and no empirical workflow trigger. The cross-backend diagnostic
must not be dispatched until that run and all of its authorized downstream jobs
are terminal.

## Claim boundary

A positive result supports **coefficient-level no-refit transfer on the existing
DLO3 source-test panel**. It does not establish:

- fresh target confirmation;
- transfer to arbitrary simulators, objects, or topologies;
- identification of true physical parameters;
- calibrated uncertainty on another backend;
- strict causal identification;
- deployment safety; or
- state of the art.

A negative result is equally informative: it would show that the correction
**procedure** is portable while the fitted discrepancy remains backend-specific.
