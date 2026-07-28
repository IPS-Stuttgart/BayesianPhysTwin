# Deform360 projected-view action-response source v3 result

## Status

The frozen projected-view source smoke completed at commit
`4f35d2cc3e8a93688a1855d822fa64683a0d913f`. It read RGB only through frame
`57` of the already-open source case and did not read a future object
observation, hidden identity, target metric, held-v8 artifact, or sealed V1
target.

The certificate **rejected**:

- measured actuator displacement: `32.441 mm`;
- independent camera views: `7`;
- required passing views: `6`;
- passing views: `0`;
- decision: `insufficient-action-aligned-response`;
- artifact ID:
  `sha256:f89640abdac14c81379133259674cfe730f3e531bd5d8be7ddf7d99070e8db96`.

No candidate belief was constructed or scored. Exact physical-baseline
fallback remains mandatory.

## What the view-space audit shows

Forward/reverse support remained stable at every update:

```text
11 / 11 / 9 / 15 / 5 / 5 / 5 identities per camera
```

Thus V3 removes the dynamic triangulation failure that eliminated V2's third
panel. It does not, however, establish enough conservative action-aligned
evidence.

Two cameras contain a strong qualitative physical-response signal:

| View | Direction cosine | Gain | Conservative lower bound | Physical RMS |
| --- | ---: | ---: | ---: | ---: |
| strongest | 0.988 | 0.375 | -0.194 | 22.462 mm |
| second | 0.955 | 0.363 | -0.295 | 18.573 mm |

Their uncertainty remains too large for the frozen positive-gain gate. Other
views either disagree directionally or see less than `0.1 mm` of identifiable
camera-tangent physical motion despite the identities being responsive in
global 3-D. The V2 global-response planner therefore does not guarantee
camera-specific observability.

## Decision

V3 does **not** justify candidate construction, a larger outcome evaluation, or
a state-of-the-art claim. Its thresholds must not be relaxed on this examined
source case.

The next technically distinct source experiment, if pursued, should select
camera/identity evidence from the sealed **projected** physical response before
opening RGB. That planner must be frozen without using these observed gains and
evaluated on different already-open source objects. It should retain the same
metric covariance, cycle-based association, residual-independent reliability,
and exact fallback.

This result also narrows the broader route: camera geometry and tracking alone
are not a sufficient safety certificate. Any candidate state update still
needs a source-calibrated, baseline-relative regret upper bound and an explicit
common-mode bias nuisance.
