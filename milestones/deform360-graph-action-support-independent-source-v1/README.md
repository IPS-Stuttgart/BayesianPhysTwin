# Deform360 graph-action-support independent source result

## Status

Frozen negative transfer result. The registered calibration outcomes and all
target panels remain sealed.

The prediction-first runner sealed an automatic frame-zero PhysTwin prediction
for every one of the 27 prospectively registered public-source episodes before
opening that episode's future object observations. All 27 seals were unique and
all 27 untouched futures were scored.

## Frozen result

| Metric | Result | Registered gate |
| --- | ---: | ---: |
| Execution-balanced future track improvement | -15.09% | >= 5% |
| Execution-balanced future Chamfer improvement | -11.67% | >= 5% |
| Late track improvement | -16.90% | >= 3% |
| Late Chamfer improvement | -12.68% | >= 3% |
| Joint future wins | 7 / 27 | >= 18 / 27 |
| Per-object joint-win fraction | 0.00-0.40 | >= 0.60 |

Every conjunctive transfer gate failed. The maximum relative degradation was
12.10x for track error and 6.77x for Chamfer, so this is not a near miss that
can be repaired by uncertainty calibration. Calibration and target evaluation
are therefore forbidden under the lock.

The complete checksummed gate is in
`artifacts/independent_source_gate.json`.

## Post-failure diagnosis

The checksummed source-only diagnosis is in
`artifacts/failure_diagnosis.json`. It is explicitly non-deployable because it
uses the 27 opened outcomes.

| Diagnostic | Future track | Future Chamfer | Joint wins |
| --- | ---: | ---: | ---: |
| Frozen alpha = 0.9 | -15.09% | -11.67% | 7 / 27 |
| Oracle fixed-physics-or-persistence switch | +7.47% | +7.41% | 7 / 27 |
| Oracle nonnegative alpha | +9.22% | +9.26% | 13 / 27 |
| Leave-one-object-out closure gate | +3.16% | +3.20% | 3 / 27 |

Thirteen episodes select exactly zero under the nonnegative alpha oracle. The
dominant problem is therefore whether the physical response should be applied,
not just its global magnitude. The closure-only gate recovers a positive
out-of-object trend but remains below the registered 5% transfer threshold and
does not identify the correct episodes reliably enough.

## Scientific interpretation

The source-discovered response is heterogeneous: seven episodes improve both
track and Chamfer, including several double-digit wins, while a small number of
episodes receive catastrophically wrong motion. A fixed 0.9 response is thus
not a transferable predictor.

The automatic episode twin also exposes a concrete model error. Its frame-zero
graph builder completes a material chain from the observed object to each
controller group even when the gripper is initially separated from the object.
The official Warp rollout then treats those anchors as persistent. This can
apply motion through a virtual attachment when real contact starts later,
changes, or never occurs.

The failure supports a narrower next hypothesis:

> infer a time-varying realized contact state from known gripper openness,
> controller/object proximity, and source-trained tactile supervision; activate
> a controller spring only while that contact is plausible; otherwise fall back
> exactly to persistence.

This is a Causal4D intervention-realization correction, not another global
blend coefficient. The 27 outcomes are now development-only for that method.
Any positive claim requires a new lock over unused Deform360 objects.

The next source-development ladder is:

1. persistence and the failed fixed-response predictor;
2. a source-trained opening-only contact transition;
3. opening plus controller/object proximity;
4. dynamic spring activation with the contact patch selected at predicted
   contact onset;
5. the same model with a conservative response-uncertainty fallback.

Only a method that clears cross-object development gates without outcome-derived
features should be frozen on a fresh set of unused objects.

## Information boundary

- Independent public-source futures read: yes, all 27 registered episodes.
- Calibration outcomes read: no.
- Target initial observations read: no.
- Target actions read: no.
- Target outcomes read: no.
- State-of-the-art claim supported: no.
