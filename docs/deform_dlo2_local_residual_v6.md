# DEFORM DLO2 Validation-Selected Local Residual v6

## Purpose

The fixed DLO1 shrinkage transferred useful mean signal to DLO2 validation but failed its preregistered worst-case cap. Version 6 changes only the correction shrinkage. It keeps the physical checkpoint, fit trajectories, ridge, causal feature definition, covariance construction, clamped-node policy, and exact fallback from v5.

The finite validation-only bank was `0.125`, `0.25`, `0.375`, and `0.5`. The locked rule selected the lowest validation L1 among arms with at least 1% mean improvement, at least 6/8 wins, and worst trajectory ratio at most 1.05. Shrinkage `0.25` was selected with 8.49% mean improvement, 7/8 wins, and worst ratio `1.0480`.

## Source Boundary

The eight DLO2 source trajectories remained unopened during selection. The source runner must verify the v5 closed result, the v6 development artifact, the exact training result, source manifest, update-6,400 checkpoint, and reproduced fitted-model hash. It then writes a source-opening seal before loading any source trajectory.

The source gate is unchanged: at least 1% mean improvement, at least 6/8 wins, worst ratio at most 1.10, and candidate mean L1 strictly below the published DLO2 reference of 9.7 mm. Failure preserves the selected physical checkpoint byte for byte. Passing authorizes a separately frozen all-training refit; it does not itself open official evaluation.

## Information Contract

The correction receives only two observed material states, the known future clamped-node action, and the physical-baseline rollout. Future free-node truth is fit or score data only. Prob4D is unused, all four clamped nodes remain exactly on the physical baseline, and both official DLO1 and DLO2 evaluation trees are read-guarded.

## Bound Development Evidence

- Parent v5 result SHA-256: `804023b09bc960cf988fb8e6ed181b4d557c826c12d7f7a7d75a4f57e8deb876`
- Validation selection SHA-256: `56365ef30f511e296ffbbb0d22001fd1bc07f655ed87656b60c23c0897a2bef1`
- Training result SHA-256: `1f8d092bc38b03f6cdd68ef38abcb7d403d914e38ba483698579deaeea8c2572`
- Selected checkpoint SHA-256: `b64affff638c9d47ca51f17bb7124cc4bd224facd1f7137b0042b7fa9037ea65`
- Expected fitted-model SHA-256: `ed4feae941ca9c293860d21f89846761ab2e5b9e6c64d97dc798d6b4db7acf10`
- DLO2 source opened during development: false
- Official evaluation read: false
