# RGBench ARCSim competence v9 result

## Decision

**Gate failed. Close the ARCSim backend without opening any RGBench
point-cloud outcome.**

Two isolated full-resolution source replays completed successfully and were
byte-identical. They preserved all 9,865 vertex identities, remained finite,
produced nontrivial motion, and required 27.12 s and 27.22 s of simulator time.
However, ARCSim's penalty handles missed the prescribed RGBench actuator
targets by as much as:

```text
21.7585 mm
```

The frozen gate allowed at most 0.01 mm. This is a factor of about 2,176 above
the limit and comparable to the benchmark accuracy differences of interest.
The candidate therefore fails the actuation contract even though its numerical
replay is deterministic and efficient.

## Frozen checks

| Check | Result |
|---|---:|
| Both replays complete | Pass |
| Byte-identical final arrays | Pass |
| Finite vertices | Pass |
| 9,865-node identity contract | Pass |
| Expected 10-step readout | Pass |
| Nontrivial mean motion | Pass, 56.211 mm |
| Runtime at most 600 s | Pass, 27.220 s max |
| Pin error at most 0.01 mm | **Fail, 21.758 mm** |

ARCSim also serialized one post-end frame beyond the explicitly indexed 0.100 s
readout. The runner ignored that later file. This quirk did not affect the gate
decision.

## Evidence boundary

- Frozen implementation:
  `703f824254c4b84e0482f69947bf6acfb2cce06d`
- Remote result:
  `/home/florianpfaff/results/rgbbench-arcsim-competence-v9-703f824`
- Gate SHA-256:
  `34fb35fdfeb75339f56e8264108a6a43e0afa6cc9168b84209fcc460786fbfe5`
- Replay array SHA-256, both runs:
  `5abc2f75e03e8528c758a542e1b417d8e1fb2c1499443dbe3c4e6765f7fd1ecc`

No point-cloud filename, point-cloud coordinate, source accuracy outcome,
calibration outcome, target outcome, or future object state was read.

## Interpretation

This closes official ARCSim 0.2.1 with its native penalty-handle interface for
the RGBench source case. It does not show that ARCSim cloth mechanics are
inaccurate; no cloth outcome was inspected. It shows that the released solver's
control interface is too compliant for a fair measured-actuation comparison at
the relevant millimetre scale.

Post-gate hard projection or handle-stiffness tuning would change the registered
method after seeing the failed diagnostic. That is not authorized under this
protocol.
