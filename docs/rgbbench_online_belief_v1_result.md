# RGBench Online-Belief V1 Result

## Status

The prospective run is closed as `predictions_incomplete`. The registered
27-case source gate was not produced, calibration was not authorized, and
calibration and target point-cloud outcomes remain unopened.

This is a public-backend executability failure, not a negative target result
for the guarded belief update.

## Runtime gate

The exact initial method was frozen at `062b900`. Its sealed
`green_tshirt/grasp/01` smoke completed, but four independently launched
brown-coat simulations segfaulted in native PyBullet before writing a
physical baseline. A serial repeat of `brown_coat/fling/01` reproduced exit
code 139, ruling out parallel execution as the primary cause.

The upstream fixed-point wrapper requests `p.GUI` unconditionally even when
visualization is disabled. A pre-outcome amendment at `fc7e412` mapped that
request to `p.DIRECT` without changing physics inputs or the belief method.
Its first gate passed exactly: the GUI and DIRECT green physical archives had
the same SHA-256,
`c62d7ffdb07f09bcd996d83084c64caa70b0d1cb547db0d3a0ea0681114a7d38`,
and every vertex, face, and target time was byte-identical.

The decisive second gate failed. The DIRECT brown replay segfaulted during
scene construction before writing a baseline. The amendment is therefore
rejected and was not used to continue the cohort.

Outcome accounting is:

- 1 ordinary sealed source prediction;
- 4 native GUI technical failures;
- 4 younger workers cancelled before a baseline was written;
- 0 retained technical-failure predictions;
- 0 calibration or target outcomes opened.

The compact evidence is
`results/sota/rgbbench_online_belief_v1/runtime_gate_result.json`.

## Sealed smoke

On the one completed source case, the static graph correction was admitted
and improved the official primary future metric by 0.73%, from 40.259 mm to
39.963 mm. Full-window error improved by 0.69%, from 31.869 mm to 31.650 mm.
The published open-loop GarmentDynamics value for this cell is 22.6 mm, so
the frozen static correction did not close the SOTA gap.

## Development lead

After the source result was opened, an explicitly exploratory linear
residual trajectory was fit from the same allowed prefix. At full scale it
reduced future error by 32.10%, to 27.335 mm. Keeping the physical prefix
unchanged would yield a full-window score of 22.296 mm, 1.35% below the
published 22.6 mm cell.

This arm is not an admissible result. It worsened the disjoint prefix
validation score by 3.33% and won zero validation frames. The useful lesson
is narrower: RGBench contains enough temporal discrepancy headroom for an
action-conditioned residual model, but its scale must be learned on
independent source actions rather than selected by weakening the gate on this
opened capture.

## Decision

Do not revise the garment split, drop the dense garments, or count technical
failures as predictions. A new RGBench protocol requires a reproducible
physical backbone that can run all locked garment meshes. Once that exists,
the strongest next method is a source-trained dynamic discrepancy

```text
c(t + h) = c(t) + h * dc(t)
```

with hierarchical shrinkage by action and garment, uncertainty that grows
with horizon and action distance, and exact fallback to the unchanged
physical rollout. The current v1 source, calibration, and target cohorts must
not be reused as a fresh confirmation of that method.
