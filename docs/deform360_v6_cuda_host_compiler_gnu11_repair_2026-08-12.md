# Deform360 v6 GNU 11 host-compiler repair - 2026-08-12

## Observed failure

Protected-main workflow run `31572805759` retained a bounded technical-failure
receipt before physical-manifest materialization or source prediction. The
preceding amendment selected `/usr/bin/gcc-12` and `/usr/bin/g++-12`, but those
executables are absent on the sole authorized runner, `workstation2`.

The retained receipt reports:

- terminal stage `build-isolated-gpu-source-runtime`;
- exit code `1`;
- `0/10` physical manifests;
- `0/100` sealed source predictions; and
- no development-suffix, v5-confirmation, v6-target, replacement, or claim
  boundary opened.

## Correction

The additive GNU 11 amendment preserves the failed GNU 12 amendment and binds
the existing CUDA 12.1.1 bootstrap to the runner's installed GNU 11.5.0 pair:

- `/usr/bin/gcc-11`, resolving to
  `/usr/bin/x86_64-linux-gnu-gcc-11`;
- `/usr/bin/g++-11`, resolving to
  `/usr/bin/x86_64-linux-gnu-g++-11`;
- exact package versions and resolved-binary SHA-256 identities; and
- the existing compile-only `nvcc --compiler-bindir` probe.

A source-independent runner probe compiled a trivial CUDA translation unit with
this pair and the checksum-pinned CUDA 12.1.1 toolkit. The production bootstrap
repeats that probe and fails closed if a path, version, binary digest, or compile
result differs. It does not use `-allow-unsupported-compiler`.

Both technical-failure and successful source receipts bind the additive repair,
selected paths, resolved binary identities, observed version, and probe state.

## Boundary

This changes only the host compiler used to build the already-declared gsplat
CUDA backend. It changes no candidate, roster, source/target split, physical or
perception model, loss, gate, covariance, fallback, or claim boundary. A new
protected-main execution must still complete the unchanged prediction-first
source protocol before any suffix or target access can be considered.
