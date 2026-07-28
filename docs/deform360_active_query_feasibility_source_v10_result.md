# Deform360 Active-Query Feasibility V10 Result

## Decision

The frozen source gate failed:

| Source result | Count |
| --- | ---: |
| Complete eight-query frame-zero budget | 4 / 8 |
| Abstention | 4 / 8 |
| Required to advance | 6 / 8 |

No tracker-provider stage, state update, hidden-identity evaluation, or fresh
target run is authorized.

## What Worked

The frame-zero depth/mask association front end was not sparse. Every case
retained between 69 and 595 graph identities with at least two selected-camera
associations. The four admitted cases then supplied all eight moving queries.
Their selected queries had exactly two-camera support in three cases and
three-camera support in one case.

This establishes that a two-view, covariance-bearing query proposal can
materialize a complete initial budget on part of the source distribution
without reading a tracker or future observation.

## Why the Gate Failed

The four abstentions retained 256, 274, 351, and 595 frame-zero association
candidates, but selected zero queries after the locked 2 mm
action-conditioned physical-motion gate. With no contact exclusion and ample
association support, the failure is attributable to the selected physical
backbone predicting no eligible response, not to a shortage of query pixels.

That matters strategically. Active query selection cannot rescue a backbone
that predicts persistence or negligible action response. Relaxing the motion
threshold would create trackable but causally uninformative points and is not
authorized by this result.

## Evidence Boundary

The pre-outcome implementation was frozen and pushed at
`73246013099541abb953af4aa9fe780e385fbf94` before the eight reports were
generated. The exact deployment passed 17 focused tests and changed-file Ruff
checks. A clean native environment passed 1,305 tests with 28 skips.

Every result used only:

- the sealed source physical rollout through frame 57;
- the graph basis and camera calibration;
- frame-zero metric depth and object masks.

No tracker output, future object observation, state correction, identity
target, future metric, V1 sealed target, or held-v8 artifact was read.

## Interpretation

V10 closes the proposed frame-zero-only active-query feasibility route under
the current selected physical backbones. It does not reject physics-guided
querying when the physical model predicts a measurable response, as four cases
passed. It shows that this planner cannot be the next general source-to-target
SOTA step by itself.

The next method should address action-response support in the physical belief
or use a separately justified causal motion trigger. It should not spend GPU
time on another tracker until a target-free source gate demonstrates that the
trigger covers the intended distribution.

Case reports and numeric archives are under
`results/sota/diagnostics/deform360_active_query_feasibility_source_v10/`.
