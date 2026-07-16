# Deform360 reusable dynamics 081 v1

This milestone freezes the source side of the first same-object reusable
PhysTwin experiment before independent calibration dynamics are scored.

## Frozen source selection

- Attempted grid: 24 official-PhysTwin parameter tuples.
- Source executions: `081-stripe-rope` episodes 1, 4, and 6.
- Selection frames: processed frames `[1, 60)`.
- Pooled tuple: spring `10000`, drag `10`, dashpot `100`.
- Single-source controls:
  - episode 1: `50000 / 3 / 100`;
  - episode 4: `30000 / 10 / 100`;
  - episode 6: `10000 / 10 / 100`.
- Eighteen tuples were finite on every source execution.
- All six `Y=80000` tuples were rejected jointly because at least one source
  execution became non-finite. The failures remain recorded in the selection
  table and were not replaced.

## Fixed-trust compatibility

The pooled tuple was evaluated with the previously source-frozen policy

```text
action response = 0.4 / controller_count
autonomous drift = 0.1
```

On source tails it achieved:

| Metric | Persistence | Pooled method | Improvement |
| --- | ---: | ---: | ---: |
| Track RMSE | 10.905 mm | 9.554 mm | 12.39% |
| Symmetric CD | 8.251 mm | 7.821 mm | 5.22% |

It won both metrics in two of three source executions. The largest per-execution
degradation was 5.89%, below the frozen 25% source sanity limit. All conjunctive
source-compatibility gates passed.

These are source-only results. They establish enough competence to run the
one-shot independent calibration on episodes 0, 2, and 8; they are not evidence
of reusable-dynamics transfer or state of the art.

## Independent calibration result

The frozen method was then scored once on episodes 0, 2, and 8. The primary
pooled, cardinality-normalized trust arm produced:

| Horizon | Metric | Persistence | Pooled method | Improvement |
| --- | --- | ---: | ---: | ---: |
| Full `[1,76)` | Track RMSE | 13.695 mm | 10.665 mm | 22.12% |
| Full `[1,76)` | Symmetric CD | 13.630 mm | 11.295 mm | 17.13% |
| Late `[51,76)` | Track RMSE | 20.996 mm | 15.662 mm | 25.40% |
| Late `[51,76)` | Symmetric CD | 20.436 mm | 16.172 mm | 20.87% |

The method jointly beat persistence in two of three executions. Its maximum
deterministic-repeat RMSE was 0.0265 mm, maximum p99 object-edge strain was
2.29%, and its rank-3-of-3 conformal radius was 42.60 mm. These registered
gates passed.

The pooled tuple matched or beat the median of the three single-source-selected
controls on both metrics in only one of three executions, below the frozen
two-execution requirement. This was the only failed gate. The pooled method
missed the median single-source control by 0.22 mm track and 0.29 mm CD on
episode 0; on episode 2 it tied track and missed CD by 0.24 mm.

Episode 8 also exposed incomplete direct actuation support: one of two selected
controller points was 39.1 mm from the reconstructed object and only one
controller spring was created. This was retained as observed.

The calibration result is therefore frozen as **negative for the pooling
claim, positive for reusable prediction versus persistence**. Episode 5 remains
sealed. Raw pooled Warp was neutral in track error (-0.24% improvement), so the
large primary gain must be attributed to the source-frozen trust/control-variate
layer rather than to the raw simulator alone. No multi-object or state-of-the-art
claim follows from this one-rope experiment.

Canonical calibration artifact:

- result SHA-256: `3a85a727f423d9822a705b38046fe0a4d96ae964e08f173636b48cdb70b89785`;
- file SHA-256: `4f7785622e376fa628068e8c4a2dfdb1e35e36c8f9c6e3599ec68e160828499c`.
