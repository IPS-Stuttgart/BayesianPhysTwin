# DEFT Cross-Branch Source Pilot: Not Promoted

## Result

The isolated native DEFT adapter passed all six synthetic qualification checks.
The subsequent locked public-training pilot completed all eight arms, but the
primary cross-branch state-transfer gate **failed**. Preserve unchanged native
DEFT on this case and the existing successful DEFORM result. Do not tune the
pilot's window, gain, observed identities, or spatial extension on these scores.

This is one BDLO1 training recording, selected by filename before decode. The
released full checkpoint has training exposure. It is source capacity evidence,
not independent confirmation, a calibrated posterior, or official benchmark/SOTA
evidence. DEFT is the branched-rod simulator; it is not the earlier DEFORM backend.

## Hidden-Child Results

All numbers are millimetres. Aggregate entries give equal weight to the two
child branches. Their seven free material identities are disjoint from the
eight later main-branch observations; duplicate roots and padding are excluded.
Every arm shares the two initial full states and prescribed future clamp motion.

| Arm | Extra point observations | Child RMSE | Coordinate L1 | Late RMSE | FDE |
|---|---:|---:|---:|---:|---:|
| Unchanged full DEFT | 0 | **28.006** | 12.808 | 32.156 | **31.571** |
| Physical-only shadow | 0 | 32.871 | 14.925 | 37.680 | 33.365 |
| Persistent readout | 8 | 30.925 | 13.002 | 39.670 | 38.703 |
| Linear-velocity readout | 8 | 49.018 | 20.468 | 66.137 | 71.422 |
| Paired physical pose | 8 | 29.414 | 12.830 | 32.873 | 33.033 |
| Paired physical pose + velocity, primary | 8 | 30.370 | 13.164 | 33.777 | 35.724 |
| Parent-only paired physical update | 8 | 28.366 | **12.703** | **31.710** | 34.323 |
| Direct full-DEFT pose + velocity update | 8 | 30.489 | 13.279 | 33.540 | 35.088 |

The primary increases hidden-child RMSE by **8.44%** versus unchanged full DEFT.
The best propagation control by that metric, the parent-only update, is still
1.29% worse. Its improvements in L1 and aggregate late RMSE do not rescue the
locked primary or establish a consistently better prediction.

| Arm | Child 1 RMSE | Child 2 RMSE |
|---|---:|---:|
| Unchanged full DEFT | 31.229 | 24.783 |
| Persistent readout | 42.954 | **18.895** |
| Primary paired physical update | 35.572 | 25.167 |
| Parent-only paired physical update | 31.854 | 24.879 |

The primary fails against the native baseline on both child RMSEs and late
horizons; it also loses to persistence on child 2. There is no significance test
or confidence interval for one recording. The 840 hidden point-time events per
arm are not independent replications.

## Interpretation

The extra observations are informative locally: persistent readout reduces RMSE
on the five disjoint hidden parent identities from 20.820 to 11.206 mm. Yet its
constant extension down each child helps one branch and harms the other. The
physical continuation does not reliably resolve that ambiguity in this pilot.

This rejects a blanket claim that the unchanged DEFORM-style sparse update
automatically transfers to unobserved branches. It does not establish that all
state estimation fails, that parent measurements can never inform a child, or
that DEFT is a poor backend. The native full model is better than its physical-
only shadow here, reinforcing the decision to preserve the stronger incumbent.
The exact role of unobserved child state, twist, geometry, and model discrepancy
is not identified by this one case.

## Verification and Provenance

- Native qualification: source `9d21fdbf96b68d66a7b796ede7cf2d66ead2868b`;
  six checks pass, with bitwise monolithic/segmented/zero-update trajectories and
  internal state, and zero synthetic clamp error. No trajectory was decoded.
- Runtime: pinned CPU, float64, one thread, Torch 2.0.1+cu118 and Theseus 0.2.1.
  This is the documented compatibility runtime, not the upstream README's
  recommended Torch 2.5.1+ environment or a reproduction of its best score.
- Pilot method: `c43fd4fa086c1625db11c3b5621fbe5255eccf17`, frozen before source
  decode. Source receipt SHA-256:
  `32746884aa39169394a1cd96cf7e26fe186b99c0a8286fd851c5dc9c912a7986`.
- Prediction barrier SHA-256:
  `6d1f16d4dc1b640fdb7edfe647c012be94695042e1c68dd272de70a4cfb5c191`.
  All eight arrays were sealed before scoring; prediction generation took 73.94 s
  on the pinned CPU runtime. This is not a benchmarked real-time claim.
- Result SHA-256:
  `e2d4ef51eb619c3ce17273c3122584d15019b22c1be031a8f68d497b89b8f305`.
- Independent metric implementation was committed before scoring at
  `ea2eca20dc5a5ae51a09bc0f99a87ce4b2b4a778`.
  It recomputes all 96 child/aggregate metrics and all 18 gate decisions from
  sealed arrays using an independent raw-identity mapping. Verification SHA-256:
  `2a5843b38540c6e2b7c32a4f79986fec9cd1ffbd9566a4dd3805e7a7de827d04`.
- Focused restart/belief regression suite: **224 passed**. Changed Python files
  pass Ruff; both implementation/runner pairs and the verifier pass focused MyPy.
- Accounting: one ordinary successful recording, zero retained technical
  failures, zero unsealable recordings; no replacement or empirical retry.

No public evaluation/test content was decoded or inspected. A source-audit Git
query did lazily cache some upstream test blobs, as recorded in the native
qualification protocol; this is not a claim that no test bytes were downloaded.
No protected DEFORM DLO3 evaluation or DLO4/DLO5, held-v8, Deform360 target, or new
physical recording was accessed. Existing DEFORM implementation and frozen
results are unchanged. No broader evaluation is authorized by this failed pilot.
Results remain local/private-paper evidence.
