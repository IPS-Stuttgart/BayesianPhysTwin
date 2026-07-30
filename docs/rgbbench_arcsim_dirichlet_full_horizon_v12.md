# RGBench ARCSim Dirichlet Full-Horizon v12

## Purpose

The v11 target-free gate established that the bound full-resolution ARCSim
backend is byte-deterministic over ten steps and follows both released actuator
trajectories with zero pin-target error. V12 tests whether those properties
survive the complete released source action before any RGBench point-cloud
filename, coordinate, or accuracy outcome is read.

This is a numerical and control qualification, not an accuracy experiment.

## Frozen Rollout

Two independent processes replay the 9,865-node green-shirt source mesh from
the same released initial pose and measured actuator trajectories. The released
horizon is 16.355 s, while the backend uses a fixed 0.01 s step. The protocol
therefore rounds upward to 16.36 s and 1,636 steps; the actuator interpolator
clamps the final 0.005 s to the released terminal pose.

The source mesh, material parameters, disabled mechanisms, patched ARCSim
source, executable, exact Dirichlet controls, and single-thread environment are
unchanged from v11.

Each replay must:

- complete all 1,636 steps with finite, identity-preserving vertices;
- produce byte-identical final arrays;
- keep the two moving pins within 0.01 mm of their targets;
- produce at least 20 mm mean cloth displacement;
- finish within the frozen five-hour per-replay ceiling.

## Information Boundary

Allowed diagnostics are limited to the released source mesh, material metadata,
actuator trajectories, fixed action timing and pin identities, replay equality,
pin error, displacement, and runtime.

The runner must not read:

- segmented point-cloud filenames or coordinates;
- source accuracy outcomes;
- calibration-garment outcomes;
- target-garment outcomes.

A pass authorizes only a separately frozen one-case source accuracy protocol.
A failure closes this ARCSim route without any point-cloud scoring. No solver,
material, timestep, collision, or control tuning is authorized after this gate.
