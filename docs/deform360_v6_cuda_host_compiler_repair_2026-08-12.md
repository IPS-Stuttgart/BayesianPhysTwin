# Deform360 v6 CUDA host-compiler repair — 2026-08-12

## Observed failure

Protected-main workflow run `31570771026` reached the isolated GPU runtime
bootstrap and failed before physical-manifest materialization or source
prediction. The checksum-pinned CUDA 12.1.1 `nvcc` rejected the runner's default
GNU compiler because its major version was later than 12.

The retained receipt reports:

- terminal stage `build-isolated-gpu-source-runtime`;
- exit code `1`;
- `0/10` physical manifests;
- `0/100` sealed source predictions; and
- no development-suffix, v5-confirmation, v6-target, replacement, or claim
  boundary opened.

## Correction

The additive amendment
`deform360_official_hub_fresh_object_session_v6_cuda_host_compiler.json`
binds the runtime to:

- `/usr/bin/gcc-12` for `CC`;
- `/usr/bin/g++-12` for `CXX`, `CUDAHOSTCXX`, and `NVCC_CCBIN`;
- exact GNU major version 12 for both compiler drivers; and
- an explicit `nvcc --compiler-bindir` probe before any source-science package
  installation or execution.

The bootstrap fails closed when either registered executable is unavailable,
does not resolve to an executable, reports another major version, or disagrees
with the other driver. It deliberately does not use
`-allow-unsupported-compiler`.

The compiler environment, repair identity, repair-file digest, and observed
compiler version are exported into the workflow environment and retained in the
runtime bootstrap log.

## Boundary

This is a technical runtime repair chained to the existing gsplat CUDA repair.
It changes no roster, source/target split, model, loss, gate, physical query,
fallback, or scientific code. It authorizes no suffix or target access and no
scientific or paper claim. A subsequent protected-main execution must still
complete the unchanged source-prediction protocol and publish its own bounded
receipt.
