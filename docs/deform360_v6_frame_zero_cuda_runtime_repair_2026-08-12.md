# Deform360 v6 frame-zero CUDA runtime repair

Date: **2026-08-12**  
Status: **pre-source-prediction technical repair; all suffix, confirmation, and target data remain closed**

## Retained failure

Protected-main source run `31532027045` completed the target-closed runtime
bootstrap, reconstructed the frozen source inventory, staged the first real
prefix with SAM2, and entered frame-zero Gaussian-splat reconstruction. The
pinned `gsplat==1.4.0` package then reported that no CUDA toolkit was available.
Its CUDA backend remained `None`, and Nerfstudio stopped at the first
rasterization call with:

```text
AttributeError: 'NoneType' object has no attribute 'CameraModelType'
```

The retained evidence is:

| Item | Value |
| --- | --- |
| Source revision | `98e20353fa80caa8a58e1885c487f3bfbfb02b93` |
| Workflow run | `31532027045`, attempt `1` |
| Artifact ID | `9117335082` |
| Artifact digest | `sha256:bb14ce5365ba3dcd46c17d52ea0c2ba36d71ec678176ef761bfe65a526e49917` |
| Receipt ID | `e8b1d22443251e2a5f14a8538c0f73f591ea10c1d625e0b6326fbb885a63a190` |
| Terminal stage | `frame-zero:026-sock-cloth-ep0007` |
| Physical manifests | `0/10` |
| Source prediction seals | `0/100` |

Every information-boundary flag remained false. No development suffix,
confirmation payload, fresh target, or target outcome was opened.

## Root cause

The primary runtime is intentionally compatible with the frozen SAM2 checkout,
which requires PyTorch 2.5.1. The gsplat 1.4.0 binary matrix does not provide a
matching precompiled wheel for that interpreter/runtime combination. The PyPI
package therefore falls back to just-in-time CUDA compilation, but the runner
provides the NVIDIA driver and CUDA-enabled PyTorch rather than the `nvcc`
toolkit required by that fallback.

Installing an unpinned toolkit into the shared runner would widen the repair
surface and make the result depend on an ambient compiler. The repair instead
uses the exact precompiled gsplat wheel for its published Python 3.10,
PyTorch 2.4, and CUDA 12.1 combination.

## Repair

The new protected-main workflow builds two isolated runtimes:

1. The **primary runtime** remains Python 3.12 with PyTorch 2.5.1+cu121 and the
   frozen SAM2 checkout. It executes inventory construction, prefix staging,
   physical-prior construction, and source-plan generation exactly as before.
2. The **frame-zero runtime** is a hermetic Python 3.10 environment with
   PyTorch 2.4.0+cu121, torchvision 0.19.0+cu121, Nerfstudio 1.1.5, and the
   official precompiled `gsplat==1.4.0+pt24cu121` wheel.

A fail-closed dispatcher routes only the exact physical-source command with one
unique `--stage frame-zero` binding to the second runtime. Every other command
uses the primary runtime. Before source execution, the workflow verifies that
the compiled gsplat backend exists and exposes `CameraModelType`; it does not
accept an import-only probe.

The activation is recorded in an immutable marker and added to the execution
receipt together with both runtime identities, package indexes, repair ID, and
repair-file digest.

## Frozen scope

The repair changes no object, episode, camera panel, RGB frame, selector, SAM2
model, physical algorithm, reconstruction settings, candidate mean, candidate
covariance, prediction horizon, loss, fallback, suffix rule, or target rule. It
authorizes only the already-registered source execution. Ten complete physical
manifests and 100 immutable source-prediction seals remain mandatory before any
development suffix may be opened.
