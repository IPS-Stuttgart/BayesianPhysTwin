# RGBench ARCSim Source Accuracy v13 Result

## Decision

**The frozen one-case source gate failed. Close this ARCSim route without
calibration or target access.**

The 27-frame prediction was generated and sealed before any source point-cloud
coordinate was read. The fresh source replay exactly reproduced the qualified
v12 final state, retained the `3.817e-13` m maximum pin error, and met the
4.967 ms timestamp-alignment limit. Scoring then opened only the already
declared `green_tshirt/fling/01` source point clouds.

## Accuracy

| Method | Real-to-sim L1 Chamfer |
|---|---:|
| Remeshed PyBullet physical baseline | 59.394 mm |
| **ARCSim Dirichlet** | **53.225 mm** |
| Best frozen cross-fitted dynamic baseline | 49.643 mm |
| Published GarmentDynamics cell | 41.900 mm |

ARCSim improves the raw physical baseline by 10.39%, but regresses 7.21%
against the strongest frozen Bayesian-PhysTwin comparator and 27.03% against
the published GarmentDynamics cell. It therefore fails two of the three
registered advancement gates.

| Horizon | ARCSim error |
|---|---:|
| Early | 40.469 mm |
| Middle | 70.170 mm |
| Late | 49.035 mm |
| Endpoint | 52.425 mm |

The middle third is the main failure. This pattern does not justify a
zero-tuning 27-case source run, much less a state-of-the-art claim.

## Provenance

- Frozen implementation: `e9f16d47ca6142c3a8306bc5452dfbcddcd7717d`
- Protocol SHA-256:
  `6025644cdabd9abd5c106d2c51a012f86f809304b9e7912873d81ca5e56d3e3b`
- Remote result:
  `/home/florianpfaff/results/rgbbench-arcsim-source-v13-e9f16d4`
- Prediction NPZ SHA-256:
  `4266332973decb9068866bec92399567fe7f2d697e3c93fff2d9a5b2bcc96bd6`
- Prediction metadata SHA-256:
  `a7c3c7bc8f945d7296bb067d74700abe8eabca3964f4765886847c1f9280fa33`
- Result SHA-256:
  `bd3d3e3f1eaf822d1ac8317a77104ba5796065eae4fcc0d0b10a824a51821082`
- Cross-host Boost 1.74 runtime-library SHA-256:
  `545886948b178af06160dfd7590ee8be108bd1e28263bf63816e4a2c43cd0a59`

The source operator ran on `gpuserver6000` from the exact pushed commit using
the same registered ARCSim executable and the exact Boost 1.74 runtime library
copied from the qualified 4090 environment. Thirty-five focused tests passed on
the execution host before simulation.

## Interpretation

This is not a failure of the control interface or numerical solver. The
full-resolution ARCSim thin shell is deterministic, stable through the entire
action, exactly actuated, and materially better than the raw public physical
baseline on this case. Its unchanged public material model nevertheless does
not close the gap to either the strongest existing Bayesian-PhysTwin correction
or the published simulator.

Per the frozen protocol, no stiffness, damping, bending, collision, timestep,
or alignment tuning is permitted after this score. Calibration and target
outcomes remain unopened. The actionable SOTA direction remains a guarded
online belief update around the best physical/action-supported backbone, not a
further public-solver substitution campaign.
