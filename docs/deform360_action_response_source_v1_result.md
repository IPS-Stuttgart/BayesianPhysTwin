# Deform360 action-response source v1 result

## Status

The frozen source-only smoke at commit
`463593be8a2604335c04b0a3d1ebe28142246413` completed without reading a
future object observation, hidden identity, target metric, held-v8 artifact, or
sealed v1 target.

The action-response certificate **rejected**:

- measured actuator displacement: `32.441 mm`;
- independent camera groups: `3`;
- passing groups: `0/3`;
- decision: `insufficient-action-aligned-response`;
- artifact ID:
  `sha256:4ff2fa9f48795f7c4cf1d4259022b8aff4664814a6ddb68a8f3a0070faf84690`.

No candidate belief was constructed or scored. The complete-belief policy
therefore remains the exact physical baseline.

## Failure localization

The valid triangulated identity counts for the three disjoint panels were:

| Prefix frame | Group 0 | Group 1 | Group 2 |
|---:|---:|---:|---:|
| 19 | 4 | 0 | 0 |
| 38 | 5 | 0 | 0 |
| 57 | 5 | 0 | 0 |

This is a frame-zero planning failure rather than evidence against the
action-response hypothesis. The inherited center planner selected identities
for global camera coverage before the cameras were split into disjoint panels.
An exhaustive target-free audit of those 16 selected identities shows that no
three-way partition of the eight cameras can provide more than `1` shared
identity in its weakest panel.

The limitation is not inherent to the source assets. Repeating the same audit
over all frame-zero graph nodes and requiring a nontrivial sealed physical
response yields a best weakest-panel support of `40` identities
(`282/127/40` across the three panels). Camera-group and identity selection
must therefore be optimized jointly before tracking.

## Consequence

V1 is retained as a safe negative control: strict grouped triangulation
abstained instead of producing a harmful update. It has no advancement power.

The next source arm may change only the target-free query plan:

1. partition cameras using frame-zero visibility and the sealed physical
   response;
2. select a balanced set of material identities with preregistered minimum
   support in every disjoint panel;
3. rerun the unchanged tracker, covariance construction, and admission
   thresholds;
4. retain exact fallback and do not inspect a future identity or loss.

A passing certificate would still not establish improvement over persistence.
That requires a separate, source-calibrated regret guard and disjoint hidden
identity evaluation.
