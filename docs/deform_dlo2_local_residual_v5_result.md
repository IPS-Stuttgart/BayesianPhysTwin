# Fresh DEFORM DLO2 Local-Residual v5 Result

## Decision

The frozen DLO2 validation-transfer gate failed its preregistered worst-case condition. The candidate improved mean validation L1 by 12.06% and won 7/8 trajectories, but one trajectory reached 1.1501 times the physical-baseline error, above the locked 1.05 maximum. The runner therefore returned the selected DLO2 physical checkpoint exactly and did not load the DLO2 source partition.

This result does not authorize an all-training refit or official DEFORM evaluation. It is a clean negative transfer decision with useful source-development headroom, not a benchmark or state-of-the-art result.

## Exact Execution

- Source commit: `8cc85de7` (`Preregister fixed-arm DEFORM DLO2 source transfer`)
- Source archive SHA-256: `409a95d2de5b07c7d893f21c44e8a619598fbfc20f45dfe07d51b1e102d59a5d`
- Protocol SHA-256: `072cbdf3c971af4c78d21b8a3c7056a4670da8df8721ab46577c8c33fff08889`
- Training result SHA-256: `1f8d092bc38b03f6cdd68ef38abcb7d403d914e38ba483698579deaeea8c2572`
- Selected checkpoint: update 6,400, SHA-256 `b64affff638c9d47ca51f17bb7124cc4bd224facd1f7137b0042b7fa9037ea65`
- Runtime: Python 3.10.12, Torch 2.0.1+cu118, CUDA 11.8
- Training root: `/home/florianpfaff/source-only/deform-dlo2-local-residual-v5/train-8cc85de7`
- Evaluation root: `/home/florianpfaff/source-only/deform-dlo2-local-residual-v5/evaluate-8cc85de7`
- Focused exact-runtime tests before execution: 46 passed
- DLO2 source partition read: false
- Official evaluation read: false

The local-residual arm was transferred without DLO2 tuning: ridge `1.0`, shrinkage `0.5`, selected only on the earlier DLO1 development study.

## Locked Validation Result

| Stage | Baseline L1 | Candidate L1 | Relative change | Wins | Worst ratio | Decision |
|---|---:|---:|---:|---:|---:|---|
| DLO2 validation (8 trajectories) | 7.9120 mm | 6.9580 mm | -12.06% | 7/8 | 1.1501 | Fail |

The frozen gate required at least 1% mean improvement, at least 6/8 wins, and a worst trajectory ratio no greater than 1.05. The first two conditions passed; the worst-case condition failed. No threshold was changed after observing the result.

Mean coordinate NEES was `0.606` and empirical coordinate coverage was `95.39%` at nominal `90%`. These values indicate conservative validation uncertainty, but they do not override the transfer-safety failure.

## Interpretation

The causal local-residual model contains substantial DLO2 signal: seven trajectories improve and aggregate error falls by nearly one millimeter. Its current global shrinkage, however, is not sufficiently selective for the remaining trajectory. That is precisely the failure mode the worst-case gate was designed to catch.

The admissible next research step is a new source-only method version that predicts when and how strongly to apply the correction using causal fit/validation evidence, while preserving byte-exact fallback. It must be preregistered before using any fresh source outcomes. Reopening this v5 decision, relaxing its 1.05 cap, or inspecting the sealed DLO2 source partition would invalidate the independent transfer test.

## Artifact Identities

- Authorization: `38696c10008d256c24fc482a475fa9c1d10185bb109d447ab0c67d701a9491ac`
- Validation transfer seal: `0c2e052ed5bbacb69b36736dc76814ca00d016e271c8cda28951e0f601f1b087`
- Frozen local-residual model: `ed4feae941ca9c293860d21f89846761ab2e5b9e6c64d97dc798d6b4db7acf10`
- Result: `804023b09bc960cf988fb8e6ed181b4d557c826c12d7f7a7d75a4f57e8deb876`
