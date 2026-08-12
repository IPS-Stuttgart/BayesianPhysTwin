# Deform360 v6 primary headless `pynput` runtime repair

The sole protected-main source execution at revision
`d98294c61f485d9f5c8d86e390bb9a6d26086bb7` retained a bounded technical
failure before producing a physical manifest or source prediction seal. The
compact source-only artifact rehashed cleanly, all suffix and target-access
flags remained false, and the official PhysTwin import stopped at
`qqtt/engine/trainer_warp.py` because `pynput` was absent.

The frozen official PhysTwin environment recipe explicitly installs `pynput`.
The registered Deform360 arm constructs `InvPhyTrainerWarp` with
`pure_inference_mode=True`; interactive keyboard control is neither selected
nor needed. This amendment therefore completes the isolated primary runtime
with exact hashes for `pynput` and its Linux dependencies, forces the package's
documented dummy backend, and probes the full frozen `InvPhyTrainerWarp` import
closure before any source case can run.

The repair does not change the cohort, candidate, physical algorithm,
optimizer, mean, covariance, loss, prediction horizon, selector, fallback,
suffix policy, or target policy. A reviewed merge to protected `main` may
produce exactly one new source-only execution. Advancement still requires all
10 physical manifests and all 100 source prediction seals; otherwise the run
is retained as another technical failure with exact fallback and closed target
boundaries.
