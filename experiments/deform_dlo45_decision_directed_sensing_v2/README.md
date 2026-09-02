# DEFORM decision-directed virtual sensing v2

This source-calibrated source-test pilot replaces the diagnostic three-collinear-
action portfolio from v1 with genuinely competing, source-fitted future-shape
actions.

No new data are collected. The 14 official evaluation trajectories of each DLO
remain absent from the runtime filesystem. For each DLO, 39 training
trajectories fit the source model, 9 disjoint training trajectories select one
sensor likelihood scale and one regret tolerance, and 8 disjoint training
trajectories form the source-test cohort.

## Task and action portfolio

The registered task scores the future 25-frame trajectories of central internal
nodes 4--7. The baseline sees the current and previous endpoint geometry and the
recorded future endpoint-action path. Eight current-prefix internal-node
readouts are masked.

Source residuals are clustered in a task-space projection. For each target
window, the local support yields:

- the exact endpoint-based physical fallback; and
- one competing future-shape correction for every represented source-response
  class.

The exact quotient certificate evaluates all compatible within-class source
hypotheses. No single latent state is selected inside a quotient class.

## Active measurements

Revealing one virtual sensor supplies an already recorded internal node's
current line-relative 3-D position and one-frame line-relative velocity. The
policies are:

- expected decision-regret reduction;
- expected full-state variance reduction;
- expected task-query variance reduction;
- fixed center-out acquisition;
- deterministic random acquisition; and
- a diagnostic prefix oracle.

All acquisition paths and actions are frozen before future internal-node
outcomes are sliced for scoring.

## Calibration

The separate source-calibration split selects one likelihood scale and one
regret tolerance from a fixed grid. A candidate is eligible only when it
produces at least 20 nonfallback calibration decisions, improves equal-
trajectory task RMSE, and keeps the harmful fraction among nonfallback
decisions at or below 5%. The objective then maximizes equal-trajectory task
RMSE improvement, followed by action coverage and lower sensing cost.

## Interpretation boundary

The experiment is source-test-only and exploratory. It tests whether action-
specific information acquisition can outperform state- or query-uncertainty
acquisition while many physical hypotheses remain plausible. It is not an
official evaluation-split result, unseen-object generalization, learned-vision
validation, continuous-control certification, deployment safety, or state of
the art.
