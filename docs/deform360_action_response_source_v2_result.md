# Deform360 action-response source v2 result

## Status

The opt-in balanced source smoke completed at commit
`9c560298c997a5935433f1906967021443f70425` without reading a future object
observation, hidden identity, target metric, held-v8 artifact, or sealed v1
target.

The certificate **rejected**:

- measured actuator displacement: `32.441 mm`;
- frame-zero panel support: `11/13/5`;
- passing groups: `0/3`;
- decision: `insufficient-action-aligned-response`;
- artifact ID:
  `sha256:f174a7b5145fc1131b1494be2574f6bb306e6bd1389a316e75f7471859cbbc95`.

No candidate belief was constructed or scored. Exact physical-baseline
fallback remains mandatory.

## What improved

The target-free balanced planner fixed the V1 query-planning defect.
Triangulated dynamic support changed from:

```text
V1: 4/0/0, 5/0/0, 5/0/0
V2: 10/7/0, 8/6/0, 9/6/0
```

The strongest panel carried ten material identities and agreed directionally
with the sealed physical response:

- direction cosine: `0.949`;
- response gain: `0.259`;
- conservative gain lower bound: `0.073`;
- observed response RMS: `8.206 mm`.

This is useful source evidence that physically supported camera motion exists,
but the lower bound remains below the frozen `0.10` gate.

## Why it still fails

The second panel disagreed with the physical response:

- direction cosine: `0.114`;
- response gain: `-0.007`;
- positive-cluster mass: `0.443`.

The third panel began with five eligible shared identities, and AllTracker
reported all five visible in both contributing cameras, but every dynamic
triangulation failed the unchanged geometric checks. Thus the remaining
failure is not frame-zero query scarcity. It is dynamic multiview geometry and
cross-panel disagreement.

## Decision

V2 does **not** justify a larger or fresh-object run. The thresholds must not be
relaxed on this examined source case.

The next credible observation-provider experiment is a source-frozen
view-space response certificate: compare AllTracker flow with the projected
physical response independently in each camera, remove camera-specific
translation nuisance, and require agreement across views before any 3-D state
update. This avoids using fragile two-view triangulation as the admission
measurement while leaving metric 3-D candidate construction and the
baseline-relative regret guard unchanged.

That experiment needs a multi-object already-open source panel. One source
case is sufficient to expose the interface failure but not to set thresholds
or authorize prospective evaluation.
