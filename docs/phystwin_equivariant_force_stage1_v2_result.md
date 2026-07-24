# Equivariant generalized-force Stage-1 result

Date: 2026-07-24

Status: failed source competence; Stage 2 is not authorized.

## Frozen execution

- deployment commit: `6245d427f339736c05e4db7231627077d2359c46`
- Stage-1 implementation SHA-256:
  `a93a272ff1b3c518591c715d9cf5f83222aba3ae6a849cef4a6e1e8bb3c4c49c`
- source protocol SHA-256:
  `1178ffe1545158225818723c700991f76d730c3627ab09644b73f2a14f53a171`
- registered folds: 3
- registered seeds per fold: 3
- source cases: 17
- target artifacts opened: false

The three folds ran on the exact CPU-validated `gpuserver4090` deployment and
were merged mechanically. Each held-out latent used only its permitted prefix.
No Stage-2 Warp rollout or historical-target access occurred.

## Result

The frozen gate required at least 10% mean held-out normalized force-RMSE
improvement, both overall and in at least two of three folds.

| Statistic | Result |
| --- | ---: |
| Fold 0 mean improvement | 0.81% |
| Fold 1 mean improvement | 2.00% |
| Fold 2 mean improvement | -0.08% |
| Overall 17-case mean | 0.97% |
| Overall median | 0.35% |
| Positive / negative cases | 12 / 5 |
| Cases reaching 10% | 0 / 17 |
| Passing folds | 0 / 3 |

The strongest case was `rope_double_hand` at 9.17%. The largest regressions
were `weird_package` (-1.92%), `single_push_rope_1` (-1.76%), and
`single_lift_sloth` (-1.05%).

The registered decision is therefore:

```text
force_target_competence_passed: false
official_warp_promotion_authorized: false
```

## Evidence

- full competence record SHA-256:
  `093fda115e99116a37967fef1a358df9fa9eaf5cedd58f8991573c1c067ac34d`
- summary SHA-256:
  `03452cff82c51d65c5c03fb0be326f7a1adeb55dd881d13de5dbce229c7c5120`
- fold-record SHA-256 values:
  - fold 0:
    `0665d3bd963181996f9381f409ccfa763684d1d034fed556545ed4767446b45e`
  - fold 1:
    `7c6cb1762565f4b605547f67cff9947ec8a0f1b7f481d4305d5f939f848c2cc5`
  - fold 2:
    `3e4f47995b0622fb57bce4b385a0fc1cbc9e284bfdd1b66b120a4f45c56be785`

The checked-in result files are
`results/sota/phystwin_equivariant_force_stage1_v2/source_competence_record.json`
and `results/sota/phystwin_equivariant_force_stage1_v2/summary.json`.

## Interpretation

The frozen low-rank equivariant generalized-force model does not transfer
inverse-dynamics targets across interactions. Its held-out predictions remain
too close to the zero-force baseline, despite valid per-node targets and a
non-saturating force scale.

This rejects this model and training protocol as the next state-of-the-art
route. It does not reject neural residual dynamics generally, nor does it
establish a trajectory-level failure, because the preregistered gate correctly
blocks the official-Warp experiment before it starts.

No architecture, optimization, threshold, or fold will be retuned against
these opened source-suffix outcomes. Subsequent work should use a separately
registered method and fresh evidence.
