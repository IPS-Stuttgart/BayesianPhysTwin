# DLO3 no-refit cross-backend coefficient transfer

- Decision: **no-refit-cross-backend-transfer-supported**
- Source trajectories: **8**
- Source panel: DLO3 train/source-test; official evaluation unopened

## Mean coordinate L1

| Method | Mean L1 (mm) |
|---|---:|
| `raw_pyelastica` | 22.8547 |
| `pyelastica_specific_candidate` | 17.7445 |
| `deform_no_refit_equal_seed_transfer` | 22.1726 |
| `deform_no_refit_seed_42` | 22.1543 |
| `deform_no_refit_seed_43` | 22.2311 |
| `deform_no_refit_seed_44` | 22.1397 |

## Primary no-refit comparison

- Equal-seed DEFORM coefficient transfer versus raw PyElastica: **2.98%** relative improvement.
- Trajectory wins/ties/losses: **8/0/0**.
- Paired trajectory-bootstrap 95% interval for candidate minus backend: **[-0.7932, -0.5666] mm**.
- Maximum candidate/backend trajectory ratio: **0.9831**.
- Improving DEFORM source models: **3/3**.

## Backend-specific reference

- PyElastica-specific refit versus raw PyElastica: **22.36%**.
- Fraction of its gain retained without coefficient refitting: **0.1334800727026542**.

## Claim boundary

Retrospective source-only DLO3 coefficient-transfer diagnostic. A positive decision supports no-refit transfer from DEFORM-fitted local residuals to sealed PyElastica source predictions; it does not establish target confirmation, arbitrary-backend transfer, zero-shot object generalization, safety, or state of the art.
