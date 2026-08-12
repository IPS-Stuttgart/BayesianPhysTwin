# Deform360 v6 primary CMA runtime repair

## Retained failure

Protected-main run `31548916577` used source revision
`942214b80c9fb257b6cd98bef079150645a9039b`. The frame-zero CUDA,
dispatcher-namespace, and fallback-config routing repairs all activated. The
first registered source case then stopped in the physical-prior stage before
any physical manifest or prediction seal was created.

The bounded artifact is `9123586004`, with digest
`sha256:8404656b614c8999748ff871e104efd8e0a128cdfa6bffc2b82cf586ada9f901`
and execution receipt
`05b00579271313b79f9c96b6ba247d9c8cb7722df4d3e1588050edc2beb287a1`.
Every suffix, confirmation, target, replacement, and claim authorization flag
remained false.

## Diagnosis

The frozen official PhysTwin checkout imports `qqtt.engine` before selecting
the gradient-based physical trainer. That package initializer imports
`qqtt/engine/cma_optimize_warp.py`, which unconditionally imports `cma`. The
official environment recipe declares `pip install cma`, but the isolated v6
primary runtime omitted that distribution and failed with
`ModuleNotFoundError: No module named 'cma'`.

This is an import dependency, not evidence that the CMA optimizer was selected.
The registered physical arm remains the same automatic PhysTwin prior using
`InvPhyTrainerWarp`.

## Repair boundary

The primary runtime now installs only the exact binary wheel
`cma-4.4.4-py3-none-any.whl`, with SHA-256
`edb6d02eb2aac2d54650f16a8f0c70711ff17445957de7c9de92ff7fd4b7ef38`,
using `--no-deps`, `--only-binary=:all:`, and `--require-hashes`. Runtime
admission imports `cma` and verifies distribution version `4.4.4`; the compact
receipt records the activation and wheel identity.

This changes no candidate, cohort, physical algorithm, optimizer selection,
mean, covariance, loss, prediction horizon, selector, fallback, suffix policy,
or target policy. It authorizes one reviewed protected-main source execution
only and does not authorize a scientific result or any outcome access.
