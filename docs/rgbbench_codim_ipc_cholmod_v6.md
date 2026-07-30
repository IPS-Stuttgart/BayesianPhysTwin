# RGBench Codim-IPC CHOLMOD qualification v6

## Purpose

Codim-IPC's single-thread Eigen backend passed the v5 target-free mechanics
gate, but each 0.1 s replay took roughly 115 s. Fixed 16-thread execution was
byte-identical but did not improve runtime. A full-horizon qualification with
that backend would therefore spend hours on a configuration already known to
be inefficient.

V6 tests Codim-IPC's existing CHOLMOD direct-solver path. This is a numerical
backend qualification, not a new physical model and not an RGBench accuracy
experiment.

## Build patch

Upstream commit `9c6cbe3` normally fetches an archived Intel MKL 2021 package
when CHOLMOD is selected. That URL no longer yields the package in the runtime
environment. The bound patch
`third_party/patches/codim_ipc_cholmod_system_blas_v6.patch`:

1. makes the Python module output directory configurable so Eigen and CHOLMOD
   builds cannot overwrite each other;
2. links the existing CHOLMOD path against system BLAS and LAPACK instead of
   the unavailable MKL archive;
3. exports a compiled `linear_solver_backend` marker.

The patch does not change the shell energy, time integrator, collision model,
Dirichlet projection, Newton solve criteria, or RGBench material parameters.
The gate verifies the marker and the exact module SHA-256 before simulation.

## Qualification sequence

The implementation is frozen before compiling the candidate module. The module
digest and patched-source digests are then bound into a separate protocol
artifact before any timed replay.

Two isolated 0.1 s source-mechanics replays must:

- complete with finite, contract-preserving vertices;
- be byte-identical;
- retain exact projected moving pins;
- produce nontrivial cloth motion;
- satisfy the frozen runtime ceiling relative to the measured Eigen control.

Only a pass authorizes a separately frozen longer target-free rollout. Source,
calibration, and target point-cloud accuracy remain sealed throughout.
