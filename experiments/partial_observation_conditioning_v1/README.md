# Partial-observation conditioning: completed source-only pilot

## Verdict

The fixed-prior conditioning mechanism is supported, but the stronger practical
Bayesian-superiority hypothesis is **not supported by this pilot**.

At identical prior means and coordinate marginal variances, the full covariance
update changes hidden-node Euclidean RMSE from **35.372495 to 26.339805 mm**
(25.54%), with 16/16 trajectory wins. The same-prior diagonal control cannot
change hidden means. A marginal-preserving sign-scrambled covariance gives
47.630391 mm. These are single-update ablations along the full observer's own
prior trajectory, not independent complete observer rollouts.

In the complete causal replay, a source-fitted deterministic regression observer
is better than the full posterior: **23.575163 versus 26.339805 mm**. A fixed
steady-state gain is essentially tied at **26.313564 mm**. Thus dynamic posterior
propagation has not demonstrated incremental practical value over the strongest
completed controls. This experiment does not establish an advantage of the
existing BayesianPhysTwin implementation or the official DEFORM checkpoint.

## Execution and provenance

- Repository: IPS-Stuttgart/BayesianPhysTwin.
- Branch: `experiment/partial-observation-conditioning-v1`.
- Completed workflow: [33985002712](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/33985002712).
- Scientific revision: `9a9a29a7b5196415b175741558fdf65b1d18b14a`.
- Runner labels: `self-hosted`, `gpuserver4090`; actual runner: `workstation1`.
- Trigger: a push changing only `.github/requests/partial-observation-conditioning-v1.json`.
- Artifact ID: `9974879726`, name `partial-observation-conditioning-v1-33985002712`.
- Artifact ZIP SHA-256: `e95db4d9e7dedf68a370537f74bc11e15abc051aa22a25c78e860880349625de`.
- Result JSON SHA-256: `157af567237523c6bb7ff9a3c69b10fa35f737479fb46e046f00ab61d8cada05`.
- Evaluator SHA-256: `8e4f51405ab751f5f95a48d8a324fe4b48d46629dcf321c88a0371002ae29920`.
- Twelve numerical/information-boundary tests passed locally and on the runner.
- An earlier workflow-validation failure, run `33984932310`, had zero jobs and no data access. Only runtime path declarations changed before the successful request; scientific code and settings were unchanged.

Compact evidence retains protocol, exact source-file hashes and rosters, source
model hashes, source-only selections, 640 case rows, result JSON, summary, logs,
and exact evaluator/tests. It contains no raw trajectory arrays or model NPZs.
The downloaded ZIP, code and method-seal hashes, source splits, all primary means,
and all seven paired bootstrap intervals were independently recomputed from the
compact artifact. That audit did not rerun raw trajectories or refit models.

## Data and information boundary

The sole empirical dataset is the canonical public DEFORM DLO4/DLO5 recording set:

```text
/mnt/seagate10tb/florianpfaff/datasets/deform/data_set
```

Only each object's 56 official **training** trajectories are used. A fixed,
filename-hash partition gives 39 fit, 9 calibration and 8 source-test recordings
per DLO: 78 fit, 18 calibration and 16 scored trajectories total. This is a new
within-pilot split of already-studied objects, hence retrospective intra-object
pilot evidence, not untouched-object confirmation. No official `eval` trajectory,
reserved external campaign, PokeFlex take, or robot experiment is opened.

Each trajectory has 500 recorded frames and 12 material nodes. Every fifth frame
is used, giving 100 observation steps. The four end nodes are always supplied.
The eight interior nodes are scored separately. An orthogonal coordinate change
is applied without ground-plane clipping. Artificial visibility masks are applied
to real measured geometry; no synthetic dynamics or injected measurement noise
enters the empirical scores. Numerical tests use synthetic fixtures only.

Initialization uses a source-fitted endpoint-to-interior prior, not the hidden
initial state. The predictor receives only endpoints and a masked observation
array, with unavailable interior coordinates replaced by NaN. Later interior
observations cannot change earlier estimates or issued forecasts. Future recorded
endpoint motion is supplied equally to every arm for the action-conditioned
forecast. Future interior positions are used only by the scorer. Both objects'
models and source-only settings are sealed before source-test deserialization.

## Shared model and comparison methods

This is a **linear dynamical surrogate**, not the released DEFORM hybrid solver:
a 24-dimensional residual around the endpoint chord, a source-fitted stable
first-order transition, endpoint-conditioned forcing and source-estimated joint
process/initial covariance. It is shared across all arms. The experiment isolates
the observation update without requiring physical-backbone retraining, but it
cannot establish a complete physical-twin contribution.

| Arm | Definition | Primary hidden-node RMSE [mm] |
|---|---|---:|
| Source-trained deterministic readout | Ridge regression of hidden state on the same current endpoints and visible coordinates | **23.575163** |
| Fixed gain | Full source-model steady-state Kalman gain, fixed separately for each visible-node subset | 26.313564 |
| Full posterior | Recursive joint covariance and Gaussian conditioning | 26.339805 |
| Frozen covariance | Condition using unchanged source initial covariance at each step | 27.792497 |
| Graph correction | Source-selected gain with linearly interpolated visible innovations | 32.305450 |
| Model only | The shared source-fitted model without interior observation updates | 72.666689 |
| Diagonal recursive control | Remove cross-coordinate covariance before every update | 31,368,073.600818 |
| Visible overwrite | Replace visible coordinates while leaving hidden predictions unchanged | 31,549,017.075138 |

The astronomical diagonal/overwrite errors are retained observer-instability
failures, particularly on DLO5, **not credible evidence for a useful superiority
claim**. They must not be used to headline a near-100% Bayesian improvement.
The positive original destruction gate is therefore insufficient. The fixed-gain
and learned-readout results govern the practical verdict.

Full/fixed/diagonal/frozen-covariance methods each receive the same three-value
source calibration budget for assumed observation noise. Graph gain and direct
regression regularization are likewise selected from three predeclared values.
There is no source-test retuning, case removal, or post-result experiment retry.
The Gaussian posterior mean also equals a deterministic quadratic MAP solution;
a unit test verifies that equivalence.

## Conditions and primary metric

The primary endpoint averages the **per-trajectory Euclidean hidden-node RMSE**
equally across four partial-observation conditions, then equally over the eight
recordings of each DLO. Frames, coordinates, visibility conditions, and 640
method/condition/trajectory rows are not independent replications.

| Condition | Full [mm] | Fixed gain [mm] | Deterministic readout [mm] |
|---|---:|---:|---:|
| Four visible interior nodes; middle segment hidden | 23.512767 | 23.515672 | **20.110200** |
| Two visible interior nodes | 21.743737 | 21.734106 | **21.515593** |
| Alternating visible halves | 17.496538 | 17.277711 | **14.664790** |
| Partial observations with two ten-step gaps | 42.606179 | 42.726766 | **38.010070** |

A fifth all-visible condition is a separate diagnostic, excluded from the
hidden-node primary metric. Full-posterior current-state error there is
0.002527 mm; visible overwrite and deterministic readout are exactly zero by
construction. Observation updates occur every five original recorded frames;
a ten-step gap spans 50 original frames.

## Paired uncertainty on differences

Differences below are full minus comparator; negative favors the full observer.
Intervals resample entire source-test trajectories separately within each DLO,
using 10,000 replicates. They are conditional on these two already-studied objects,
not object-generalization intervals. The two destructive primary contrasts also
have 97.5% intervals in the raw evidence. Remaining 95% comparisons are reported
as exploratory contrasts, not a multiplicity-adjusted family of discoveries.

| Comparator | Difference [mm] | 95% paired trajectory interval [mm] | Full wins / 16 |
|---|---:|---|---:|
| Deterministic readout | +2.764642 | [+2.035630, +3.442462] | 2 |
| Fixed gain | +0.026241 | [-0.023059, +0.078326] | 7 |
| Frozen covariance | -1.452692 | [-1.763063, -1.134376] | 16 |
| Graph correction | -5.965645 | [-6.949442, -5.068163] | 16 |
| Same-prior diagonal, single-update audit | -9.032690 | [-10.165765, -7.883806] | 16 |
| Same-prior scrambled, single-update audit | -21.290587 | [-23.487267, -19.117264] | 16 |

The full observer is **11.73% worse than the deterministic readout** in aggregate.
It is not meaningfully better than a fixed gain.

## Genuine future prediction

Issued forecasts consume no subsequent interior observations. Errors refer to
nodes hidden at the forecast origin, using the same condition/trajectory
aggregation. These are original recorded-frame horizons, not observation steps.

| Horizon [frames] | Full [mm] | Fixed gain [mm] | Deterministic readout [mm] | Model only [mm] |
|---|---:|---:|---:|---:|
| 5 | 35.562662 | 35.485467 | **32.457469** | 72.793747 |
| 25 | 78.991861 | 78.953654 | 78.087646 | **73.546395** |
| 50 | 85.670279 | 85.663557 | 85.270918 | **73.625575** |

No general future-prediction advantage is established. In particular, the coarse
surrogate and assimilated state do not provide better long-horizon forecasts
than the model-only arm. This is not an evaluation of the released physical solver.

## Uncertainty and paper boundary

Full-posterior coordinate coverage is 92.47% at nominal 90%, with mean normalized
coordinate NEES 0.8550 and mean full coordinate interval width 45.63 mm. These
are within-pilot diagnostics, not joint-field or universal calibration evidence.

The supported statement is:

> Joint covariance can transfer a visible measurement into a beneficial hidden
> geometry correction at a fixed prior on these real recorded trajectories.

The stronger statement is not supported:

> Maintaining the full posterior gives better reconstruction and prediction than
> comparably trained deterministic observers on this pilot.

There is no claim of official DEFORM superiority, BayesianPhysTwin end-to-end
superiority, physical-parameter recovery, real-camera robustness, unseen-object
transfer, or robot safety. The current result is retained rather than repaired by
choosing favorable source-test settings.
