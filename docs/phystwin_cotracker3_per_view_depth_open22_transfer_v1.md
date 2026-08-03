# CoTracker3 per-view RGB-D state-update transfer protocol

Date: 2026-08-03

Status: locked before evaluating the 19 non-development cases.

## Question

Camera-only triangulation can be geometrically coherent and still carry a
shared metric bias. This diagnostic tests a narrower observation path: retain
each CoTracker3 camera row separately, lift it through that camera's raw depth,
model shared and camera-specific bias explicitly, and admit only physical
response modes that remain identifiable after removing the nuisance subspace.

The method is not a new tracker and does not treat dense pixels or cameras as
independent evidence. It is an opt-in Bayesian state update around the fixed
selected PhysTwin/MatPhys replay, with bit-exact fallback to that replay.

## Frozen method

Only causal prefix evidence is available. Perception reliability uses tracker
quality, forward/backward consistency, and local depth variation; it never
uses the residual against the physical prediction. The innovation enters once
through the robust likelihood.

Within a view, effective information is capped. Across views with unknown
correlation, equal-weight covariance intersection is used. A low-rank physical
response basis is restricted against shared spatial and per-camera bias modes.
An inferred direction larger than the physical guard is radially shrunk to the
tighter of 20 mm and twice the observed physical-response magnitude. The cap is
never widened.

Correction scales `0.25`, `0.5`, and `1.0` are selected from prefix Chamfer
error only, subject to no prefix regression. Manual trajectories are excluded
from fitting and selection. Any failed support, identifiability, update, or
selection gate returns the unchanged physical baseline exactly.

## Development evidence

The cap behavior was selected on three already-open sloth interactions. The
frozen selector changed their equal-case future mean as follows:

| Metric | Physical baseline | Per-view update | Change |
| --- | ---: | ---: | ---: |
| Chamfer distance | 11.968 mm | 11.468 mm | -4.18% |
| Manual-track error | 19.834 mm | 18.606 mm | -6.19% |

The update improved track error in all three cases. It improved Chamfer in the
two double-action cases and regressed `single_lift_sloth` slightly at the
prefix-selected full scale. This is development evidence, not transfer.

## Transfer gate

The transfer cohort is the other 19 released PhysTwin cases. No setting may be
changed after their outcomes are computed. Advancement requires all of:

1. at least 3% equal-case Chamfer improvement;
2. at least 3% equal-case manual-track improvement;
3. at least 12 of 19 cases improving both metrics;
4. no case-metric regression above 10%; and
5. exact fallback and the manual-selection prohibition remaining valid.

Passing authorizes only a source-only posterior calibration audit and a frozen
factorial test with the existing graph-persistence baseline. It does not
authorize a fresh target evaluation by itself. Failure stops this method family
without tuning its rank, cap, scales, or selector on the 19 outcomes.

## Claim boundary

All 22 interactions have been examined by earlier Bayesian-PhysTwin studies.
This experiment is therefore retrospective source transfer evidence. Future
RGB, depth, CoTracker3, VGGT, and manual trajectories do not form predictions.
Manual trajectories are opened only after prediction for scoring. No held-v8
or sealed target artifact is authorized.

The state coefficient covariance has implementation controls, but predictive
coverage after nonlinear rollout is not established here. A separately locked
source calibration audit is mandatory before any independent evaluation.
