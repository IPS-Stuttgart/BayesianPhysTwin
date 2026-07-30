# RGBench ARCSim Dirichlet Competence v11 Result

## Decision

**The target-free competence gate passed. No RGBench point-cloud outcome was
opened.**

Two isolated full-resolution replays completed, preserved all 9,865 vertex
identities, remained finite, moved nontrivially, and produced byte-identical
final arrays. Initializing the two declared kinematic handles before gravity
relaxation eliminated the v10 reference offset:

```text
maximum pin-target error: 0.0 mm
```

The pass authorizes only a separately frozen target-free full-horizon
qualification. It does not establish predictive accuracy.

## Frozen Checks

| Check | Result |
|---|---:|
| Both replays complete | Pass |
| Byte-identical final arrays | Pass |
| Finite vertices | Pass |
| 9,865-node identity contract | Pass |
| Expected 10-step readout | Pass |
| Nontrivial mean motion | Pass, 54.967 mm |
| Runtime at most 600 s | Pass, 27.270 s max |
| Pin error at most 0.01 mm | Pass, 0.0 mm |

## Provenance

- Frozen implementation: `707aa2533a8c77292cb8acd5985c6c077dfa0756`
- Remote result:
  `/home/florianpfaff/results/rgbbench-arcsim-dirichlet-competence-v11-707aa25`
- Gate SHA-256:
  `5857d5a1347e18b15e3bda54bac6d1fbe3274d959f549a76b41b84290634d6af`
- Replay array SHA-256, both runs:
  `c8aeaafa72d63bbb899117b02b4978a3cab940ffa613705dbe3d17b71fe761a1`
- ARCSim executable SHA-256:
  `04723de854ed50d39b9b06762bb109fc706c74d6e86029fefb767adff47db31a`

No segmented point-cloud filename, point-cloud coordinate, source accuracy
outcome, calibration outcome, target outcome, or future object state was read.

## Interpretation

V11 establishes that the bound ARCSim backend can deterministically evolve the
full-resolution source cloth while honoring the released measured actuator
trajectory exactly over the short target-free gate. This repairs the control
contract rejected by v9 and v10. Numerical stability and determinism over the
complete 16.355 s action remain untested and are the next registered gate.
