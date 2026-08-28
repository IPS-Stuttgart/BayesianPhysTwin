# Constraint-separated readout: source result

**Decision: do not advance.** The fixed complementary split worsens the successful
paired update on the already-open DLO2 source object. All 14 predictions sealed
normally; 13 trajectories contribute to the metrics after the original design
case is excluded. There are zero technical fallbacks and zero missing cases.

## Matched results

All rows use the same frozen native/readout backbone. Corrections use the same
eight permitted prefix identities; only disjoint hidden identities are scored.
These are equal-trajectory means in millimetres, not Chamfer distance.

| Arm | Coordinate L1 | Point RMSE | Late point RMSE |
|---|---:|---:|---:|
| Unchanged incumbent | 10.673 | 25.614 | 27.524 |
| Existing paired state update | **9.594** | **23.066** | **27.089** |
| Persistent sparse readout | 13.943 | 33.000 | 37.157 |
| Fixed half blend | 10.801 | 25.817 | 30.272 |
| Tangent-only ablation | 9.628 | 23.113 | 27.091 |
| Complementary constraint split, primary | 10.320 | 24.319 | 28.905 |

The primary adds **1.253 mm / 5.43%** RMSE over paired, with whole-trajectory
bootstrap difference interval **[0.483, 2.133] mm**. It wins jointly in L1/RMSE
on only **1/13** trajectories. Late RMSE worsens **6.71%**. Only the ordinary
prediction-count check passes; all six performance checks fail. The tangent-only
ablation also slightly worsens RMSE (+0.20%) and cannot rescue the primary.
The 10,000 bootstrap resamples describe this opened single-object sample; they
are not independent-object confirmation or a population claim.

## What the diagnostic says

Across the 13 analysis trajectories, a mean **99.608%** of the original paired
correction's squared norm is already in the local constraint tangent space.
Projection changes little. The primary adds a persistent normal remainder with
mean coordinate RMS **3.432 mm**; it hurts rather than explaining useful error.
The largest linear constraint certificate residual is below 3.5e-17 m.

This rejects the fixed projected-paired-plus-persistent-normal readout rule on
this source object. It does not establish that all residuals are physical or that
observation bias is absent. It does not test a nonlinear feasible-state restart,
learned constraint metric, or material parameter change. No such extension is
authorized by this failed source gate. The existing successful paired update,
upstream native simulator, checkpoint, readout, and old evidence remain intact.

## Verification and provenance

- Frozen implementation: `4756cac0c0c8f3c2094905b55f78fff69a28c886`.
- Source lock ID: `081540c3e3f1ce58cdeb9af25e3efa333f602cc030aacf85c912f2d99a327226`.
- Prediction seal ID: `90ffb3b8b303ddfb6908ddf0f687faf4d4173df754487f6a5d0541bf1ead8d8a`.
- Result ID: `239d63202d6511f13cedb1f71ba4139b1c5190531a8e67a9915d1915c0ff3e11`.
- Result file SHA-256: `c3175ada9633905a8b48b0e74f1e50b3a8cfc91f35a7f2ce67f8137a117f88e1`.
- Second-arithmetic ID: `4d771cbe7787ebd598a04f34dc6ad0c10fa48c80ea9e77e4890b60b1978a830f`.

Pre-run verification: 494 DEFORM tests, Ruff, focused MyPy, and diff checks
passed. Five additional synthetic verifier tests pass. The second implementation
uses pivoted QR instead of production SVD and matches all 1,680 projected
prediction frames within **2.23e-16 m**. It verifies 1,008 case/aggregate metrics,
four trajectory-bootstrap contrasts, seven gate checks, exact comparator bytes,
and the committed source/prediction/result chain. This is a second arithmetic
implementation by the same agent, not independent human review.

The single CPU-only screen reused saved public-data forecasts. It used no new
recordings, native rollouts, GPU, robot execution, DLO1/DLO3 transfer, DLO4/DLO5,
official DLO3 evaluation, held-v8, or reserved Deform360/Causal4D data. No main
merge, public push, method promotion, or SOTA claim was made. Full evidence is
preserved in the local user archive; compact JSON is under
`results/source/deform_constraint_split_source_v1/`.
