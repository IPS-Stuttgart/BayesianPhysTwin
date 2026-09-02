# DLO3 cross-backend one-scalar transport

- Decision: **cross-backend-shared-residual-geometry-supported**
- Interpretation: retrospective complete-trajectory cross-validation

## Mean coordinate L1

| Method | Mean L1 (mm) |
|---|---:|
| `raw_pyelastica` | 22.8547 |
| `direct_equal_seed_no_refit` | 22.1726 |
| `leave_one_trajectory_out_one_scalar` | 22.3405 |
| `pyelastica_specific_high_dimensional_refit` | 17.7445 |

## Claim ladder

- Exact equal-seed no-refit point transfer: **True**
- Cross-validated one-scalar point transfer: **True**
- Directional alignment: **True**
- Shared residual geometry: **True**

## Registered scalar gate

- Relative improvement over raw PyElastica: **2.25%**
- Wins/ties/losses: **6/0/2**
- Maximum trajectory ratio: **1.0894**
- Fixed-fold paired bootstrap interval, candidate minus raw: **[-1.1660, 0.2866] mm**

## Residual geometry

- Positive trajectory alignments: **8/8**
- Median alignment cosine: **0.3707**
- Fold-scalar minimum/median/maximum: **2.1149 / 2.4654 / 3.3128**

## Direct-transfer reference

- Direct no-refit relative improvement: **2.98%**

## Claim boundary

A positive result supports a shared cross-backend residual direction whose amplitude can be recalibrated with one scalar fitted on other complete source trajectories. It is weaker than exact no-refit coefficient transfer and is not fresh target confirmation, arbitrary-backend transfer, zero-shot object generalization, safety, or state of the art.
