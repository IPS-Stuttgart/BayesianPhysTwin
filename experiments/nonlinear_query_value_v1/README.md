# Same-mean nonlinear deformation-query test

This retrospective experiment tests the previously proposed readout
`E[||A X||^2] = ||A mean||^2 + trace(A covariance A^T)` against the
point plug-in and source-fitted controls. It does not retrain a simulator,
change coordinate predictions, collect physical data, or use reserved cohorts.

## Fixed public panel

Use the completed DEFORM DLO4/DLO5 parent run 33361441865. The successful
inventory 33984475829 binds its cache bytes and manifests. Eight source-test
trajectories per DLO fit readout controls; all 14 evaluation trajectories per
DLO are scored over 498 forecast frames. The independent loader must reproduce
the original coordinate-L1 means before a result is admitted.

The parent is a retrained DEFORM physics-plus-learned-residual hybrid, not
bare rod physics or the authors' released pretrained checkpoint. Source
control errors come from the parent 39-trajectory fit, whereas evaluation
means come from the parent all-56 refit. This distribution change limits the
comparison and is explicitly retained in every result.

## Frozen queries and readouts

There are 21 squared distances between nonadjacent free markers and six
second-difference bending statistics. Bending is a geometric statistic, not
physical energy. Thirteen queries (marker gap at least four, or odd bend
center) form the held-out-query panel. The field-level empirical covariance
controls may use full source geometry; query-specific signed corrections do
not fit held-out query outcomes. The ridge predictor's initial feature is the
first predicted geometry, not an additional observation.

Nine readouts preserve the same coordinate mean: point plug-in; native
within-marker posterior; a new source-coupled posterior extension; global
centered empirical covariance; horizon-centered empirical covariance;
sign-symmetric residual second moments; source query bias; source
query/horizon bias; and source query ridge correction.

Native covariance is 3x3 within each marker. It has no cross-marker blocks.
The source-coupled arm estimates block-normalized correlation and uses fixed
0.5 identity shrinkage. It is an empirical extension, not a retained joint
Bayesian posterior. For these rotationally invariant quadratic queries,
within-marker coordinate off-diagonals alone cannot alter the trace readout.

## Scoring and decision

Construct and seal all query predictions for both DLOs before loading either
evaluation future. Primary query RMSE uses source-only family normalization.
Retain raw query RMSE in square millimeters per complete trajectory as well.
The two posterior arms must improve at least 1% in EACH DLO for BOTH query
families and have negative upper paired-bootstrap MSE-difference bounds
against the point readout. Bootstrap 10,000 times, seed 20260906, resampling
whole trajectories within the two fixed DLOs. Strong-control superiority
additionally requires negative upper bounds against all six declared
empirical/deterministic readout controls. No selected windows or target tuning.

These are two fixed real objects, not independent draws sufficient to establish
arbitrary-object generalization. The study is retrospective, not untouched
confirmation. Quadratic moments cannot establish a uniquely Bayesian or
higher-order distribution advantage.

## Execution

PR #936 retains the earlier path-only failure and successful inventory. The
publication retry adds the numerical core, scoring driver, and 12 synthetic
software tests. A change to `request.json` with phase `evaluate` starts the
branch-scoped workflow on `gpuserver4090`; implementation-only edits do not
request empirical execution. Tests must pass before the real-data step.
The workflow retains results and failures alike. A successful workflow by
itself is not a positive hypothesis result; inspect `result.json` decisions.

No main-branch merge or scientific claim is authorized by this experiment.
