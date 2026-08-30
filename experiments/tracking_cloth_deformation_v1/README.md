# Tracking Cloth Deformation: shake-to-twist evaluation pilot

A maintained workflow for the user-installed **real public motion-capture data**
from Coltraro, Borras, Alberich-Carraminana and Torras (2025),
*Tracking cloth deformation: A novel dataset for closing the sim-to-real gap for
robotic cloth manipulation learning*, DOI `10.1177/02783649251317617`.
Dataset: <https://zenodo.org/records/14644526>.

This is an executable, explicitly limited **physical-baseline qualification and
public-data pilot**, not a reproduction of PhysTwin or clothilde-sim, not a
paper-ready validation of the complete BayesianPhysTwin API, and not another
request for new physical recordings. It changes no historical result or claim.

## Run on GitHub

Open **Actions -> Tracking Cloth Deformation evaluation -> Run workflow**.
The evaluation job requires labels `[self-hosted, Linux, X64, gpuserver6000]`.
The intended server is workstation2; routing follows the user-specified label,
not the earlier gpuserver4090 installation note. The default read-only cache is:

```text
/home/github-runner/.cache/datasets/tracking-cloth-deformation-v1-zenodo-14644526
```

The modes are:

| Mode | Data access | Outputs |
| --- | --- | --- |
| `inventory` | Archive integrity/hashes and file names; no numeric trajectories | Verified roster, licensing record, provenance |
| `source_only` (default) | 32 shaking recordings only | Four-fold source scores, parameter weights, empirical guards, variance calibration |
| `evaluate` | Source fitting, then target initialization/corners; all 32 predictions sealed before target free-marker scoring | Per-record/per-specimen scores, paired contrasts, coverage/width, fallback audit |

Choose `source_only` first. Review source model competence and the initialization
geometry/units assumptions before explicitly choosing `evaluate`. The latter
repeats the same source fit from the same versioned protocol, publishes its
prediction-seal artifact, and only then runs a separate target-scoring process.
A failed preparation or upload prevents scoring. A changed source model,
protocol, implementation, dataset or prediction artifact invalidates the seal.
All runs remain pilots: prior public outcome exposure is unknown, and rerunning
already-scored targets never creates fresh confirmation.

The cache is never downloaded, extracted, modified, or chmod'ed. A private
scratch directory and isolated Python environment are created in `RUNNER_TEMP`.
Only NumPy and CPU are required. Four CPU workers are the default. Pull requests
run **synthetic tests on GitHub-hosted runners only**; they never access the
self-hosted dataset. No runner/organization secret or repository-write permission
is needed. Concurrency does not cancel an active evidence run.

## Frozen v1 design

`protocol.json` fixes the following before target numerical use:

* Exact Zenodo record and published archive MD5
  `b4868b702f8a42b2ea1069d0f1a3b8f6`; 120 extracted CSVs must byte-match the ZIP.
* The complete free-hanging factorial: 4 materials x 2 sizes x 2 speeds x
  2 grasp modes x 2 motions. All 32 shaking recordings are sources and all 32
  twisting recordings are targets. The other 56 collision recordings remain
  unused, not opportunistic replacements.
* A 1-second all-marker initialization prefix and a 5-second scored forecast.
  Data are regularly subsampled from 120 Hz to 30 Hz; dynamics use eight
  integration substeps. No time alignment is fitted to target outcomes.
* The future measured trajectories of the **two driven corners are supplied as
  prescribed boundary conditions**, as in the dataset's simulator use case.
  They are not logged robot commands and are not scored. No future free-marker
  coordinate enters fitting, state initialization, model selection or prediction.
* Four-fold leave-one-speed/grasp-recording-out source fitting within each
  material-size specimen. A candidate guard is fixed from these source folds,
  not selected from twisting scores.

CSV parsing follows the paper's numeric `Frame, Time, X1,Y1,Z1,...` layout and
preserves missing-marker masks. Only source initialization geometry is used to
resolve metre/centimetre/millimetre scale; all source units must agree, and that
scale is frozen for targets. The pilot assumes an initially near-vertical regular
marker mesh: A2 has five rows of four markers, A3 four rows of three. Geometry
is ordered from the initial frame, with the upper-row endpoints as driven
corners. This is **not asserted to reproduce the supplied MATLAB ordering**.
Unsupported or ambiguous initialization fails the run, with no dropped cases or
outcome-dependent remapping. The four supplied MATLAB readers are hashed but not
executed. Prefix and corner gaps use causal carry-forward, never interpolation
from future free-marker observations. Missing free-marker ground truth is omitted
with the same mask for every arm and its count is reported.

## What the pilot actually computes

The physical backend is a small equal-marker-mass 3-D spring mesh with structural,
shear and two-hop bending springs, gravity, viscous damping, symplectic
integration and prescribed corner positions. Its rest lengths come from the
initial frame. The fixed bank contains stiffness-per-mass values
`[100, 400, 1600]` and damping-per-mass values `[0.5, 2, 8]`. This is a transparent
qualification baseline, not a high-fidelity material/contact model.

The seven arms are:

| Arm | Definition |
| --- | --- |
| `persistence` | Hold the last permitted prefix free-marker position |
| `nominal_physics` | Nominal `(400,2)` spring model from initial state, no prefix-end reset |
| `last_residual` | Same nominal forecast plus its last prefix residual, outside simulator state |
| `nominal_state_injection` | Nominal model reset to prefix-end positions and backward-only velocity |
| `map_physics` | Source-selected maximum-weight parameter member with the same reset |
| `bayesian_physics` | Source-weighted parameter-bank average with the same reset |
| `guarded_bayesian_physics` | Complete candidate mean/variance or exact nominal mean/variance |

The source model weights are an explicit **generalized Bayesian/Gibbs** update:
`w(k) proportional to exp(-sum_record MSE(record,k)/(2*temperature))`.
Each source recording contributes one normalized loss, not thousands of
independent marker likelihoods. Temperature is the source-only median best-member
MSE, floored at `1 mm` squared. These are not claimed to be calibrated physical
parameter posterior probabilities. Parameter uncertainty contributes the
between-member variance; three source-only out-of-fold horizon bins supply a
residual variance floor. Scoring uses a diagonal moment-matched Gaussian, not a
claim of calibrated joint trajectory covariance.

The empirical source guard accepts a specimen only when its out-of-fold
candidate mean RMSE beats **both** nominal physics and last-residual by at least
1%, and none of its four source folds regresses versus nominal physics. This is
a conservative source selection rule, **not a finite-sample deployment safety
certificate**. The unguarded controls are always reported. A rejected candidate
reuses the exact nominal mean and variance, and scoring verifies equality.

## Results and interpretation

Primary endpoint: free-marker Euclidean trajectory RMSE in millimetres, averaged
within each recording, then equally over its four speed/grasp conditions and
then over eight material-size specimens. This is **not** the dataset paper's
mass-matrix norm; no unverified marker mass quadrature is invented.

Supporting endpoints are mean marker Euclidean error, coordinate Gaussian NLL,
90% marginal coverage **with interval width**, missing scored-marker counts,
accepted/rejected recordings, harmful accepted records, worst-specimen regret,
and exact fallback violations. Comparisons include the strong last-residual and
MAP controls, not just the nominal simulator.

Paired bootstrap intervals resample the eight specimens. A four-material
cluster sensitivity analysis is also reported because sizes may not constitute
independent draws of material properties. These are small-sample, exploratory,
non-simultaneous intervals, not a multi-comparison confirmation certificate.
Frames, marker coordinates and the 32 recordings are not treated as independent
physical specimens. No case is silently removed from the registered factorial.

Artifacts contain `protocol.json`, dataset/code hashes, `source_fit.json`,
`source_scores.csv`, `prediction_seal.json`, `target_scores.csv`,
`specimen_scores.csv`, `metrics.json`, `run_manifest.json`, `report.md`, and the
included license. Prediction trajectory arrays stay in private runner scratch
and are **not uploaded**, as are all raw recordings and the ZIP. Reports are
retained for 90 days, not forever; a paper claim needs a separate durable evidence
intake after scientific review. A technical failure emits `failure.json` and no
complete scientific decision. Green unit tests are synthetic software evidence.

## License conflict

The user-retained Zenodo metadata says CC BY-SA 4.0, whereas the included
`License.txt` says **CC BY-NC-SA 4.0**. The stricter included noncommercial policy
is retained pending author clarification. The archive, metadata and originals
remain untouched in the cache; the exact license and hashes accompany output.
This workflow does not resolve the conflict or grant commercial/redistribution
rights. Do not relicense the dataset as the repository's code license.

## Local equivalent

From the repository root, in an environment with the pinned requirements:

```bash
python -m experiments.tracking_cloth_deformation_v1.run \
  --dataset-root /home/github-runner/.cache/datasets/tracking-cloth-deformation-v1-zenodo-14644526 \
  --output /tmp/tracking-cloth-source-v1 --stage source --workers 4
```

For an explicitly authorized pilot, use `--stage predict` in a **fresh output
directory**, preserve the complete prediction seal, and then run `--stage score`
against that same directory. The workflow additionally uploads the seal before
scoring. Commands never revise `claims.json` or manuscript prose.

## Workflow lifecycle and evidence classification

Classification: **operational prerequisite and scored diagnostic** for the
untested external real-cloth shake-to-twist question, specifically authorized by
the user after installing this dataset. The existing Cloth Sim2Real workflow is
bound to a different Zenodo release, simulator, cohort and historical evidence;
repurposing it would silently change its scientific contract. This one maintained
entry point provides inventory, source qualification and sealed scoring, with
experiment logic in versioned Python. No one-shot bootstrap workflow is added.
