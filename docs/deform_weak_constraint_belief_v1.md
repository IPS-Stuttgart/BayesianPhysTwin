# Weak-constraint prefix belief: opened-data development protocol

## Question and scope

Does allowing model error between prefix measurements improve the full native
endpoint state, rather than fitting all observations with one initial-state error?
This is a new, separately frozen development hypothesis after the forecast-aware
query selector failed its primary gate. Neither that selector nor its outcomes are
retuned. All existing DEFORM checkpoints, learned readouts, results, and source
rosters remain immutable. Only already-open DLO1/DLO2/DLO3 data are admitted.
DLO4/DLO5, official DLO3 evaluation, held-v8, physical Causal4D data, and protected
Deform360 targets remain excluded. No new recordings are needed.

DEFORM already alternates prediction with measurement correction (Appendix C of
[the original paper](https://arxiv.org/html/2406.05931v2)). Periodic correction is
therefore not a novelty claim. Weak-constraint smoothing and Gaussian calibration
are also established ideas. The experiment tests their useful combination with
complete native rod-state propagation, a preserved learned readout, and honest
same-mean uncertainty controls. This is not a claimed new inference principle.

## Fixed method

Four material identities (2, 4, 6, 8) are measured at prediction frames 25, 33, 41,
49. Hidden identities (3, 5, 7, 9) are evaluated over frames 50 through 169. Raw
dataset indices are prediction indices plus two. Future clamp motion is supplied
under the unchanged parent action contract; future free-node observations are not.

All arms use the same frame-25 physical prior: 12 position coefficients (10 mm
standard deviation), 12 velocity coefficients (100 mm/s), and a shared three-axis
observation translation (5 mm). Independent measurement noise is fixed at 1 mm.
The shared bias is one nuisance variable across all measurements, not independent
noise at each point. It is marginalized, never added to the physical future.

The strong arm has no process increments. The primary weak arm also has 12
velocity-increment coefficients at each of frames 33, 41, 49 (50 mm/s standard
deviation). Each increment has a linked position change of 0.04 s times its velocity
change, corresponding to half the 80 ms measurement interval. These are effective
state discrepancies, not identified forces or material parameters. The eight-query
secondary arms observe only frames 41 and 49; weak_8 admits only the frame-49 process
increment. Both retain the same frame-25 prior to isolate the added process channel.

The four-knot material basis has exactly zero displacement at prescribed clamps.
Responses to impulses are computed by symmetric native replays at +/-0.1 prior
standard deviation, before measurement values are read. The already sealed 24
frame-25 responses are reused byte-for-byte; only the 36 new process responses are
computed. Responses before an impulse must be zero. The linear Gaussian prefix
posterior retains state/process/bias correlations. Its physical posterior mean is
injected sequentially into the real native prefix rollout, retaining material-frame,
twist, previous-position, and velocity state. A single gain bounds the sum of nodal
injection magnitudes to 30 mm and 300 mm/s. This is a guarded local approximation,
not an exact nonlinear posterior or a globally monotone covariance guarantee.

The forecast is incumbent + updated native continuation - nominal continuation.
Zero increments return the exact original incumbent array. The previous successful
eight-query paired forecast is loaded and hash checked without recomputation or
replacement. No forecast-aware sensing policy or future-dependent arm selection is
used. Clean observations only are scored here; earlier noisy-query results remain
separate evidence and are not inherited as validation of this method.

## Matched controls and point gate

The nine fixed arms are incumbent, previous_paired_8, ols_physical_16,
ols_readout_16, periodic_pose_16, strong_8, weak_8, strong_16, and weak_16.
OLS fits a straight line to all four prefix residuals, evaluates its position and
velocity at frame 49, uses the same endpoint magnitude bounds, and either propagates
that native state or extrapolates it at readout. The periodic control corrects only
the four observed native positions at each measurement; velocity is unchanged and
the learned readout is preserved. Its observations are not interpolated to hidden
identities. It is a DEFORM-style baseline, not a new method.

On each of DLO1 and DLO3 the primary weak_16 must improve RMSE by at least 2% over
both previous_paired_8 and ols_physical_16; improve coordinate L1 and point RMSE over
all matched controls and the previous paired arm; win jointly on at least 5/8
trajectories against the previous paired arm; avoid worsening late RMSE versus the
incumbent; and keep every trajectory's RMSE ratio to the incumbent <=1.05.
These are development effect gates, not population significance tests. Secondary
budget eight cannot rescue a failed primary gate. RMSE/L1 are not Chamfer distance.

## Predictive uncertainty and comparators

Each strong/weak arm exports marginal 3x3 covariance in m^2 by propagating the joint
physical posterior through the native tangent responses, multiplying by the squared
mean guard gain, and adding a fixed (3 mm)^2 I model-error floor. Bias is excluded
from future physical readout. These are approximate predictive moments. No joint
independence across points, exact posterior, or raw calibration is claimed.

After all 30 predictions are sealed, only the 13 non-design DLO2 trajectories fit
three horizon scales. The primary scale is mean marginal NEES/3, averaging within
trajectory and then across trajectories. Identical fitting is applied to shaped
weak_16 covariance, isotropic covariance around exactly the same weak_16 mean, and
isotropic covariance around previous_paired_8. Isotropic raw covariance is (3 mm)^2 I.
Three fixed 40-frame horizon blocks are used. There is no target scale fitting.

An explicitly secondary conformal wrapper uses each source trajectory's 90th
percentile marginal Mahalanobis score (higher quantile), followed by outer rank
ceil((13+1)*0.9)=13, the maximum of 13 scores. Its scale is that score divided by
chi-square_3(0.9). The resulting ellipsoids are evaluated alongside Gaussian NLL
under a declared Gaussian density with that scale. Conformal sets do not inherently
define a density. This score construction does not imply simultaneous coverage of
all future points, and cross-object exchangeability is not established.

The calibration artifact is sealed before DLO1/DLO3 scoring. Marginal Gaussian NLL,
3D NEES, 90% ellipsoid coverage/volume, and geometric-mean full axis width are
averaged over events within trajectory, then trajectories within object. Point
metrics use the parent whole-trajectory bootstrap. Uncertainty contrasts use 10,000
whole-trajectory paired bootstrap draws with seed 260834. With only two opened
transfer objects, these are conditional descriptive intervals, not confirmation.

The uncertainty gate requires, on each transfer object, the shaped primary moment
calibration to have a 95% upper NLL-difference bound below zero against same-mean
isotropic calibration, coverage in [0.80,0.98], and no larger mean ellipsoid volume.
The total development gate requires both the point and uncertainty gates. The
conformal secondary cannot rescue a failure. A success would justify a separately
registered fresh test, not automatically authorize it or establish official SOTA.

## Execution and custody

Freeze clean source and protocol; verify the already-open archives and old response
seal; prepare all native responses and no-op controls; seal the response barrier;
reveal only the fixed prefix identities; seal every candidate/control prediction;
validate the complete 30-case barrier; fit and seal DLO2-only calibration; score the
29 non-design cases; independently recompute inference, covariance, metrics, and
selected native continuations. One complete run, no outcome-driven retry, no
replacement, and no public release of outcome bundles. Retain technical failures
explicitly and block advancement rather than dropping a case. Local/private-paper
artifacts are permitted. No existing result or frozen metric is rewritten.
