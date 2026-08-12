# Deform360 v6 official PhysTwin Python 3.10 runtime repair

The sole protected-main source execution at revision
`64a62c014e6e5217fc2d55970005da76b13fc31e` retained a bounded technical
failure while building the isolated runtimes. It produced zero physical
manifests and zero source prediction seals. The compact artifact rehashed
cleanly, and every suffix, confirmation, and target-access flag remained
false. The import stopped because the official PhysTwin trainer requires its
vendored `simple-knn` CUDA extension, which was absent.

The pinned official PhysTwin environment is a Python 3.10, Torch 2.4, CUDA
12.1 stack. It explicitly installs PyTorch3D, `diff-gaussian-rasterization`,
`simple-knn`, Warp, Kornia, CMA, and `pynput`. The earlier workflow attempted
the physical-prior import in the Python 3.12, Torch 2.5 primary runtime. This
repair instead routes the `physical-prior` stage, along with the already
isolated `frame-zero` stage, through the upstream-compatible Python 3.10,
Torch 2.4 runtime.

All downloaded additions are versioned and hash-pinned. PyTorch3D uses the
official Python 3.10/CUDA 12.1/Torch 2.4 wheel. The two Gaussian extensions are
built from exact source trees in the pinned PhysTwin checkout, copied to a
temporary build directory so the checkout remains immutable. The runner's
CUDA 12.9 compiler needs `NVCC_PREPEND_FLAGS=-include cstdint` for the frozen
rasterizer source; this changes compiler preprocessing only and does not edit
the source tree. The receipt records source-tree identities, generated wheel
hashes, build flags, import probes, and whether the physical-prior dispatcher
route was actually activated.

This is a runtime repair only. It changes no object, cohort, model, optimizer,
association, mean, covariance, loss, prediction horizon, selector, fallback,
suffix policy, or target policy. A reviewed merge to protected `main` may
produce exactly one new source-only execution. Advancement still requires all
10 physical manifests and all 100 source prediction seals. Any lesser outcome
is retained as a technical failure with the registered exact fallback and
closed target boundaries.
