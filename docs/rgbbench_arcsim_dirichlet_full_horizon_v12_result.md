# RGBench ARCSim Dirichlet Full-Horizon v12 Result

## Decision

**The target-free full-horizon qualification passed. No RGBench point-cloud
outcome was opened.**

Two isolated full-resolution replays completed all 1,636 steps, preserved all
9,865 vertex identities, remained finite, followed the two measured actuator
trajectories essentially exactly, and produced byte-identical final arrays.

This pass authorizes only a separately frozen one-case source-accuracy
protocol. It is evidence of numerical and control competence, not predictive
accuracy.

## Frozen Checks

| Check | Result |
|---|---:|
| Both full-horizon replays complete | Pass |
| Byte-identical final arrays | Pass |
| Finite vertices | Pass |
| 9,865-node identity contract | Pass |
| Expected 1,636-step horizon | Pass |
| Mean cloth displacement at least 20 mm | Pass, 462.209 mm |
| Runtime at most five hours | Pass, 1,496.143 s max |
| Pin error at most 0.01 mm | Pass, `3.817e-10` mm |

## Provenance

- Frozen implementation: `01bb3639cf67e7a3cca11319b24024757e32f969`
- Remote result:
  `/home/florianpfaff/results/rgbbench-arcsim-dirichlet-full-horizon-v12-01bb363`
- Protocol SHA-256:
  `f5c57253115550ca1cdd59ccacd95af1278c07d73374d96ce9377c2bb36ee9c2`
- Gate SHA-256:
  `c5cbdd5c00a45fa9ad77606a25a08f2d35fb675718bf7573ad674ee2dd61ee43`
- Replay array SHA-256, both runs:
  `a99ecaff4efd601f708123d0bd21f4ee18c5b684f0fde66ab2197f1ae55503f4`
- ARCSim executable SHA-256:
  `04723de854ed50d39b9b06762bb109fc706c74d6e86029fefb767adff47db31a`

The first direct script invocation stopped before simulation because the
repository root was absent from Python's module search path. It created no
output root. The unchanged frozen implementation was then invoked as its Python
module with `PYTHONPATH=src:.`; this is the qualified run recorded above.

No segmented point-cloud filename, point-cloud coordinate, source accuracy
outcome, calibration outcome, target outcome, or future object state was read.

## Interpretation

Unlike the prior Codim-IPC route, ARCSim did not stall under the complete
released fling. Unlike the native ARCSim penalty interface, the registered
Dirichlet boundary followed the known actuation at the relevant millimetre
scale. The remaining question is empirical: whether this thin-shell rollout is
more accurate than the current RGBench physical baseline on one already
declared source case. That comparison must be specified and committed before
opening the source point cloud.
