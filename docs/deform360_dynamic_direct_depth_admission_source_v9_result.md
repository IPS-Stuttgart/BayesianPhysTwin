# Dynamic Direct-Depth Admission V9 Source Result

## Decision

**The frozen V9 source gate fails at 0/7 admitted cases and is closed.**
The registered gate required at least four admissions. No candidate state
update was constructed, and no future identity, future metric, future object
observation, V1 sealed target, or held-v8 artifact was read.

This is a target-free provider/admission result. It is not an accuracy result
for a direct-depth state update, since no case reached that stage.

## Frozen Evidence

- Repository revision:
  `019ca9b3dc0143c7cba963025a9280f170ebfa37`
- Protocol configuration SHA-256:
  `4d746521878af293370e1bb978c7fba911490608b166094cdb93c6fc33e9f8e2`
- Locked source cases: 7
- Required admissions: 4
- Observed admissions: 0
- Registered endpoint pairs per case: `(13, 19)`, `(32, 38)`, `(51, 57)`
- Maximum RGB-D/mask frame read: 57

The complete reports and their hashes are bound in
`results/sota/diagnostics/deform360_dynamic_direct_depth_admission_source_v9/summary.json`.

## Admission Outcome

The 21 registered case/endpoint branches produced:

| Target-free branch outcome | Count |
| --- | ---: |
| Motion-stratified frame-zero query budget incomplete | 20 |
| Insufficient active support after the registered filters | 1 |
| Admitted | 0 |

Six cases could not construct the frozen nine-active/three-sentinel query
schedule at any endpoint. In the remaining case, the first endpoint had raw
support for all nine active identities and all three sentinels, but no active
identity remained usable by the action-response certificate after the
registered association, reliability, and sentinel-debiasing path. Its later
two endpoint schedules were also incomplete.

## Interpretation

V9 answers a narrower question than V8. V8 showed that a coherent direct-depth
update can still harm an almost static future. V9 asked whether a frozen,
action-aligned and common-bias-aware certificate could identify suitable
dynamic prefixes broadly enough to justify constructing updates.

It could not. The dominant failure occurs before state inference: the source
episodes do not provide enough identities satisfying the simultaneous
motion-stratified, multiview, and sentinel schedule. The one geometrically
complete endpoint also failed to supply usable action-response evidence after
the frozen filters.

This does not show that a correctly admitted direct-depth update would fail.
It shows that this particular carrier and admission contract is not viable on
the locked source panel. Relaxing its query budget, view support, endpoint
schedule, or action-response thresholds on these opened cases would be
post-gate tuning and is not permitted.

## Consequence

No candidate construction, baseline-relative regret guard, or disjoint hidden
scoring should be implemented for V9. Any future direct-depth attempt must be a
new method family with a new source panel and a carrier whose admissibility is
established before an outcome protocol is frozen. The stronger guarded online
belief route remains separate, and its independently owned prospective
evaluation boundaries remain unchanged.
