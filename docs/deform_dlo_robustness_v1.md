# DEFORM DLO robustness v1

This protocol transfers the already fixed DLO2 recipe to DLO3, which has not
been used for BayesianPhysTwin development. DLO4 and DLO5 are unopened reserves
and cannot replace DLO3. The upstream repository contains its authors' own
training logs for some DLOs; those files are not project development evidence
and are not read by this study.

Before any DLO3 trajectory payload is read, the protocol fixes the physical
training budget, seed-42 primary candidate, seeds 43 and 44 stability audit,
ridge `1.0`, shrinkage `0.25`, causal feature contract, exact physical fallback,
and all reporting operators. Prob4D is unused. A domain-separated hash of each
training basename assigns 39 fit, 9 calibration, and 8 source-test trajectories.
The source-test payload is opened only after predictions are sealed.

The source gate requires at least 1% improvement over the identically trained
physical checkpoint, at least 6/8 wins, worst-case ratio at most 1.10, and mean
L1 below 7.7 mm. At least two of three fixed seeds must pass the registered
stability criteria. Only then may an all-56 refit and one-shot DLO3 evaluation
be prepared. DLO3 evaluation cannot select a seed, feature set, covariance,
solver setting, backend, case, or retry.

The mechanism audit separates the physical backbone, action conditioning,
local coordinate frame, intercept-only correction, shrinkage, and persistence
backbone. The compute control spends the measured residual-fit wall time on
additional official DEFORM updates using the same schedule continuation. The
all-56 refit repeats that timing operation, seals both the registered
update-6400 checkpoint and the resulting compute-matched checkpoint, and rolls
both through the independent dry run and the one-shot evaluation. Its report is
descriptive: it has no pass field and cannot select the candidate, replace the
registered physical baseline, or affect the primary gate. A compute-control
failure blocks readiness in the dry run; after the one-shot target read, a
technical failure is retained without retry while the primary sealed result
remains scorable. The solver audit evaluates 5/10/20 PBD iterations and joint
bend/twist multipliers of 0.9/1.0/1.1 without selecting among them.

Backend portability is a separately gated PyElastica 1.0.0 Cosserat-rod arm.
Its finite parameter bank is selected on fit trajectories only; DLO3 target
access is allowed for that arm only if its calibration and source-test gate
passes. Failure leaves the official DEFORM candidate unchanged.

The selected PyElastica parameters, full residual-covariance model, and
calibration are sealed before source-test scoring and rehashed at all-train and
readiness. If the source gate passes, the independent dry run must execute the
backend path and the one-shot evaluator produces its predictions during the
same target read as the primary arm. The backend report is descriptive and
cannot select or alter the primary result. A target-specific backend runtime
failure is retained without retry while the already-sealed primary arm remains
scorable; a dry-run backend failure prevents readiness. When that source gate
authorizes carryover, both evaluator invocations must receive the frozen
PyElastica checkout through `--pyelastica-root`; otherwise the argument is not
needed and the backend path remains exactly unexecuted.

The same downstream authorization rehashes each source compute-matched
checkpoint, every mechanism-model archive and prediction arm, and all six
solver/material sensitivity prediction pairs. Readiness separately rehashes
the all-56 registered and compute-matched checkpoints, recomputes the exact
ceiling timing rule, checks the deterministic schedule continuation and dry-run
prediction seal, and records that target selection remains false. A JSON score
without its exact sealed NPZ and parent lineage is therefore insufficient to
authorize all-train or target use.

The Bayesian audit leaves the point mean unchanged and adds full 3x3
coordinate covariance from trajectory-clustered coefficient and residual
uncertainty. Nine calibration trajectories set the 90% scale by the maximum
trajectory score. The seven frozen arms are current conservative diagonal,
shrinkage-propagated diagonal, coefficient-only, residual-only,
pooled-isotropic, uncalibrated full covariance, and calibrated full covariance.
Every covariance is constructed and sealed before scoring, every arm is
required to preserve the point mean exactly, and none may be selected from
source-test or target outcomes. Temporal independence is not claimed.

The completeness requirement is enforced again at every downstream boundary.
The stability operator rehashes each seed's prediction seal and NPZ, requires
all seven named covariance arrays, checks their shapes, finite values, clamped
support, positive-definite internal blocks, and calibrated scaling, and then
validates all seven metric records. The all-train authorization independently
rechecks the primary seed and requires the three-seed verification receipt.
Readiness similarly rehashes and inspects the independent evaluator dry run.
An omitted arm, a changed point mean, outcome-based distribution selection, or
a stale artifact therefore prevents all-train or target authorization. A
verifier-only source revision may follow the frozen seed runner revision, but
both revisions and their artifact hashes must be bound; this does not authorize
retraining, result replacement, or any target access.

Outcome-bearing compact artifacts belong only in the private
`BayesianPhysTwin-Paper` repository. The public repository retains executable
code, the frozen protocol, and target-blind provenance receipts.

## Pre-seal runtime failure and replacement boundary

The first replacement execution at source commit `bd5cfb148` completed the
registered 6,400 updates for seeds 42 and 43 and wrote both method seals. Both
processes then failed before prediction sealing because the intercept-only
diagnostic's valid zero-width normalization matrices were passed through a
validator that required non-empty arrays. Source-panel processing had begun,
but no prediction archive, source result, source score, target access, or
reserve access was produced. The target-blind receipt is
`results/sota/deform_dlo3_robustness_v2/preseal_runtime_failure.json`.

The correction permits finite `(internal_nodes, 0)` normalization matrices
only for the already declared zero-feature arm. It does not change the physical
checkpoint, local-residual point method, feature subsets, shrinkage, covariance
construction, gate, or target policy. Every mechanism predictor is now executed
on calibration inputs before the method seal and before source-panel
processing. The failed roots remain immutable. Any completion from their exact
method seals or any replacement execution requires a separately versioned,
explicit authorization; it is not an automatic retry. DLO4 and DLO5 remain
unopened reserves and cannot replace DLO3.

The pending recovery record is
`configs/sota/deform_dlo3_method_seal_recovery_v1.json`. It binds the two
method seals, compute-match records, failure logs, source manifest, failure
receipt, and calibration-only preflight by SHA-256. Its only permitted action
is artifact validation: `source_completion_authorized` is false. The separate
recovery runner cannot train, refit, continue a checkpoint, select a seed, or
substitute a source case. A future completion would require changing the
decision coherently in a new source revision, would use one fixed empty output
root per seed, would bind and verify the exact authorized implementation
revision and source archive before any write, and would seal the recovered
predictions before scoring. No such completion has been authorized or
executed.

The artifact-only validator subsequently passed for both exact parent method
seals. The compact receipt is
`results/sota/deform_dlo3_robustness_v2/method_seal_recovery_validation.json`;
it records that no source-test payload was deserialized, opened, or scored.
The recovery decision remains pending.

On 2026-08-19 the user explicitly authorized exactly one source-only
completion each for seeds 42 and 43. The separately versioned authorization is
`configs/sota/deform_dlo3_method_seal_completion_authorization_v1.json`. It
binds merged implementation revision `68feea8a`, source archive SHA-256
`111ac9b2c2d74976277a8aba1b52663788e109ec67b796e98e619c83919e56f7`,
and the original immutable method-seal lineage. It does not authorize
retraining, refitting, checkpoint continuation, seed 44, official DLO3
evaluation, DLO4/DLO5, or held-v8 access.

After both exact completions were independently rehashed and frozen, the
registered third stability seed was separately authorized in
`configs/sota/deform_dlo3_seed44_source_execution_authorization_v1.json`. It
permits one preflight, one smoke, and one production source execution for seed
44 from exact implementation revision `da487c26` and archive SHA-256
`46fc314b89510bb4ff3e8eba4848baccbc05963842de558252243aa225bae862`.
The authorization file SHA-256 is
`886586b15da2552ae247811a7cae379b036700bc511542e02ce274ce8bcc99d4`.
The predecessor outcomes are represented publicly by hashes only. This does
not authorize a retry, method change, all-56 fit, official DLO3 evaluation,
DLO4/DLO5 access, or held-v8 access.

Protocol: `configs/sota/deform_dlo_robustness_v1.json`.
