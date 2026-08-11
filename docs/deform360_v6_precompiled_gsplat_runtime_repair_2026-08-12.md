# Deform360 v6 precompiled gsplat runtime repair

## Scope

Protected-main workflow run `31532027045` completed the previously repaired
Deform360 dependency bootstrap, but stopped during the first frame-zero
reconstruction before producing a physical manifest or source prediction seal.
The retained source-only log records `gsplat: No CUDA toolkit found` followed by
an unavailable CUDA backend. No development suffix, confirmation payload,
fresh-target payload, or target outcome was opened.

This is a runtime defect, not scientific evidence. PyPI gsplat `1.4.0` compiles
its CUDA extension on first use, while the admitted runner provides a CUDA GPU
runtime but no discoverable `nvcc` toolkit. Merely importing the Python package
therefore did not prove that Splatfacto could execute.

## Correction

The source workflow now uses the same core tuple already frozen by the public
v5.2 endpoint-processing lock:

- Python `3.10`;
- Torch `2.4.0+cu121` and torchvision `0.19.0+cu121`;
- CUDA runtime `12.1`;
- gsplat `1.4.0` from the official `pt24cu121` Linux wheel.

The runtime is created without inherited system site packages. The core CUDA
dependencies are version-locked, and the official gsplat wheel is checked by
both byte count and SHA-256 before installation. The workflow then verifies the
installed extension hash, loads `CameraModelType.PINHOLE`, and executes a tiny
GPU rasterization with a backward pass. Source execution remains impossible
unless that preflight succeeds.

This supersedes the earlier exception for an inherited PyRecEst/NumPy conflict:
PyRecEst is not installed or used by the isolated runtime, and a complete
`pip check` must now pass without exceptions.

## Scientific boundary

The reconstruction code, model size, source cohort, camera panel, Prob4D input
role, loss, gates, and exact-fallback behavior are unchanged. The repair does
not authorize a source result or any suffix or target access. A reviewed merge
to protected `main` may create exactly one new source-only execution. One
hundred immutable source prediction seals remain mandatory before later gates
can run.
