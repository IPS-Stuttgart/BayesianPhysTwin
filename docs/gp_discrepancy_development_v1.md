# Recording-specific GP discrepancy development

Owner: issue #945. Classification: prospective method and retrospective
source-development diagnostic. This is outside the stable wheel and inference
APIs. No existing estimator, public claim, frozen protocol, or result is changed.

## Model

The experimental correction augments the existing ridge predictor, not bare
physics. In fixed graph-mode coordinates it uses

`residual_s(z,t) = g(z) + b_s(t) + epsilon`,

with covariance

`K((z,s,t),(z',s',t')) = shared_variance * Matern32(z,z')
 + 1[s=s'] * session_variance * Matern32(t,t')`.

Observation noise is an explicit positive diagonal variance. Different recording
IDs have independent execution effects, but share the learned systematic
function. A new recording's execution uncertainty does not vanish when the
shared function becomes well known. A previously observed recording can be
conditioned at new times; reusing an already conditioned recording/time pair is
rejected. Output modes are conditionally independent in this first version.

Features and outputs are standardized using fit recordings only. Feature distance
is normalized by the square root of feature dimension. The declared output-scale
floor is part of the configuration. Conditioning uses Cholesky without added
jitter, covariance eigenvalue clipping, pseudo-inverses, or numerical retries.
The dense row and joint-dimension limits are explicit. This is a bounded exact
prototype, not a scalable inducing-point implementation.

The DLO adapter uses the existing causal feature builder, its action/gravity
frame, and the existing native per-node ridge implementation. Mode corrections
are learned in that canonical frame and rotated back before global-coordinate
error scoring. Dirichlet chain modes leave clamped vertices exactly unchanged.
The correction is a predictive readout discrepancy, not proof of a physically
reachable state correction or identification of material parameters.

## Separation of questions

The mean comparison reports the native hybrid, existing ridge, ridge plus GP,
and a source-selected mean. The GP mean is used only when its mean L1 improves
on the separate selection recordings. Rejection returns the existing ridge
prediction exactly; this is not a new complete-physical-belief routing claim.

The covariance comparison attaches all covariance alternatives to **one identical
selected mean**. Alternatives are raw GP, independently scaled GP, its
per-frame-block counterpart, a per-frame-sign temporal control, a calibrated
diagonal, and an empirical low-rank residual-second-moment baseline. Removing
cross-time blocks or applying per-frame signs preserves every per-frame block,
so marginal coverage and width cannot explain differences in joint scores.

The empirical low-rank comparator is a newly fitted DLO-compatible control. It
is **not** the exact frozen Deform360 dependence-study model. A win against it
must not be described as beating that earlier implementation.

Four disjoint complete-recording roles are used: fit, select, calibrate, score.
Model selection uses select only. Covariance scale, residual second moments and
the deformation-contrast threshold use calibrate only. Tests replace every score
outcome and verify identical predictive means, model choice, scales and threshold.
Bootstrap units are recordings, never coordinates or frames.

Scores include global-coordinate L1, graph-mode joint Gaussian NLL per dimension,
normalized NEES, graph-mode marginal coverage and width, and a Brier score for a
source-thresholded end-minus-start deformation contrast. Graph-mode calibration
must not be relabelled full-state calibration. The native DLO development export
scores 24 fixed forecast times, not the complete official evaluation operator.

## Controlled execution

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_gp_discrepancy_v1.py tests/test_gp_discrepancy_study_v1.py
OPENBLAS_NUM_THREADS=1 python scripts/science/run_gp_discrepancy_development_v1.py \
  --synthetic --output-dir outputs/gp-discrepancy-controlled
```

The checked-in protocol fixes five seeds and two arms. The generator combines
nonlinear bias, OU and oscillatory execution errors; it is not a draw from the
fitted Matérn model. The second arm has a hidden score-time regime shift. Both
arms and all failures are retained. These are controlled mechanism results only.

The maintained `GP discrepancy development` Actions workflow runs these tests
and controlled studies on hosted runners with read-only permissions. It does not
access physical datasets, GPUs, private checkpoint paths, or official targets.

## Native source-data execution

After review, use the existing DEFORM source compute environment with the pinned
DLO1 parent artifacts. The exporter does not train a new DEFORM checkpoint:

```bash
PYTHONPATH=src python scripts/science/export_deform_gp_development_v1.py \
  --parent-protocol configs/sota/deform_dlo_local_residual_v4.json \
  --longrun-result /path/to/pinned/longrun_result.json \
  --source-manifest /path/to/pinned/source_manifest.json \
  --upstream-root /path/to/pinned/DEFORM \
  --output-dir outputs/gp-dlo1-source-export

OPENBLAS_NUM_THREADS=1 python scripts/science/run_gp_discrepancy_development_v1.py \
  --archive outputs/gp-dlo1-source-export/development.npz \
  --manifest outputs/gp-dlo1-source-export/manifest.json \
  --output-dir outputs/gp-dlo1-development
```

The parent checkpoint/result/manifest identities are checked. Only DLO1/train is
allowed. Original validation recordings are split deterministically into
selection and calibration. Previously inspected DLO1 source-test recordings
remain retrospective development. DLO1/eval and all DLO2-DLO5 data are guarded
against reads. No terminal or sealed cohort is reopened or promoted.

Native-ridge hyperparameters use the original candidate bank, selected on the
new selection subset; this is a development refit of the native comparator, not
a byte-identical reproduction of an earlier frozen result. Only a verified
native run can establish exporter/runtime compatibility or real-data value.

Outputs contain source/runtime/input identities, individual recording scores,
all alternatives, predicted-array hashes and failure receipts. Paper-facing
results and interpretation belong in `FlorianPfaff/BayesianPhysTwin-Paper`.
