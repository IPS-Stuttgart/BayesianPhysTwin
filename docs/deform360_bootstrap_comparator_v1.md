# Deform360 source-residual bootstrap comparator v1

This experiment extends the completed 92-object, same-mean and coordinate-marginal-matched Deform360 dependence study with one source-only nonparametric comparator.

For each registered scalar query, the comparator uses leave-one-source-episode-out residuals, gives every source episode equal mass, recenters the empirical distribution to the exact frozen target mean, and rescales it to the source-calibrated query variance of the full low-rank arm. The comparison therefore holds the predictive point mean and query variance fixed while changing distributional shape and the route by which query-event probabilities are obtained.

The workflow must reproduce the archived three-arm scientific result exactly before the comparator is scored. Complete physical objects are the bootstrap units. Event Brier score and finite execute-versus-fallback loss are co-primary endpoints under a Bonferroni familywise-95% rule. Event log loss and standardized CRPS are secondary.

The experiment reuses an already opened public Deform360 target cohort. It is retrospective mechanism evidence, not fresh confirmation, absolute calibration, physical-state identification, deployment safety, or a state-of-the-art claim. A positive or negative scientific outcome does not determine workflow success; only lineage, information-boundary, parity, and completeness violations fail the job.
