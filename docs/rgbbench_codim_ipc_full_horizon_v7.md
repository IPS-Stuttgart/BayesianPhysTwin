# RGBench Codim-IPC full-horizon qualification v7

## Purpose

The v6 target-free gate established that the full-resolution CHOLMOD backend is
byte-deterministic over ten steps, exactly follows the two projected pin
trajectories, and is 27.64% faster than the frozen Eigen control. V7 tests
whether those properties survive the complete released source action before
any RGBench point-cloud coordinate or accuracy outcome is read.

This is a numerical and control qualification, not an accuracy experiment.

## Frozen rollout

Two independent processes replay the 9,865-node green-shirt source mesh from
the same released initial pose and actuator trajectories. The released horizon
is 16.355 s, while Codim-IPC uses a fixed 0.01 s step. The protocol therefore
rounds upward to 16.36 s and 1,636 steps. The existing causal trajectory
interpolator clamps the final 0.005 s to the released terminal actuator pose.

Each replay must:

- complete all 1,636 steps with finite, contract-preserving vertices;
- produce byte-identical final arrays;
- keep the projected moving pins within `1e-10` m of their targets;
- produce at least 20 mm mean cloth displacement;
- finish within the frozen five-hour per-replay ceiling.

The protocol binds the upstream commits, source mesh and trajectories, material
parameters, exact-control and CHOLMOD patches, patched source tree, CMake cache,
compiled module, and BLAS/LAPACK dependencies.

## Information boundary

Allowed inputs and diagnostics are limited to the released source mesh,
material metadata, actuator trajectories, fixed action timing and pin
identities, replay equality, pin error, displacement, and runtime.

The runner must not read:

- segmented point-cloud filenames or coordinates;
- source accuracy outcomes;
- calibration-garment outcomes;
- target-garment outcomes.

A pass authorizes only a separately frozen one-case source accuracy protocol.
A failure closes this full-horizon Codim-IPC arm without any accuracy run.
