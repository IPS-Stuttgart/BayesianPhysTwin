# DEFORM post-hoc adapter controls v1

This experiment tests the focused public-data paper claim after the protected
DLO4/DLO5 target study has completed. It does not collect data and does not
retrain the 6,400-update DEFORM backbone.

For each of DLO4 and DLO5, the runner reloads the exact all-train checkpoint and
recomputes its predictions on the 56 public training trajectories. It then
compares the already sealed 14-trajectory target predictions against:

- the registered DEFORM hybrid baseline;
- the compute-matched continuation retained by the parent run;
- two simple source-residual averages and a time-only ridge model;
- removal of explicit action features;
- removal of baseline velocity, acceleration, and curvature features;
- replacement of the initial action-aligned frame by global coordinates; and
- deterministic source-data curves at 1, 2, 4, 8, 16, 32, and 56 trajectories.

The source subsets are selected by a frozen hash ordering and never by target
outcomes. Complete trajectories are the statistical units. The DLO4/DLO5 target
outcomes were already opened by parent workflow `33361441865`; consequently this
is a retrospective post-open falsification and efficiency study, not an
additional prospective confirmation.

## Execution

The workflow is triggered only by adding
`.github/requests/deform-dlo-posthoc-controls-v1.json` on branch
`science/deform-posthoc-adapter-controls-v1-20260903`. The hosted job checks that
the trigger commit changed only that request. The empirical job runs on
`gpuserver4090`, whose local parent-run cache and verified DEFORM dataset are
required.

The workflow writes one JSON result per DLO, an aggregate JSON record, complete
trajectory CSVs, a Markdown summary, environment/custody records, and execution
logs. Raw trajectories and prediction arrays are not uploaded.

## Claim boundary

A positive result can establish that the registered lightweight residual adapter
beats both the frozen DEFORM hybrid and an equal-wall-time continuation on the
already evaluated DLO4/DLO5 operators, and can show whether simple residuals or
specific feature groups explain the gain. It cannot establish unseen-object
transfer, a uniquely physical state correction, fresh confirmation, deployment
safety, or a benefit from active robot probing.
