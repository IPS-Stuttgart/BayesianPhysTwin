# Deform360 action-conditioned real-data evidence

## Result

The nonlinear Bayesian action ensemble now has two positive real-data results
using future robot trajectories as registered intervention inputs and future
tactile measurements only for scoring.

| Cohort | Objects | Persistence | Bayesian action ensemble | Relative change | Object W/T/L | 95% object bootstrap |
|---|---:|---:|---:|---:|---:|---:|
| Development | 14 | 0.65102001 | 0.61359914 | -5.75% | 14/0/0 | [-0.047283629, -0.027711315] |
| Reserved confirmation | 4 | 0.82952018 | 0.79542845 | -4.11% | 4/0/0 | [-0.068708788, -0.0033560784] |

The relation-breaking control also passed in both cohorts:

- development ensemble versus action-shuffled control: **-3.79%**, with
  **14/14** object wins;
- reserved confirmation ensemble versus action-shuffled control: **-2.77%**,
  with **4/4** object wins.

## Confirmation discipline

The v3 protocol, nonlinear feature map, ridge grid, ensemble weighting,
source-only bias correction, guard and target-selection rule were frozen at
revision `25ba91c021124569c4dcf84c66eda5ec088868e0`, before the predecessor v2
target result was available. A metadata-only readiness run then identified
exactly four reserved objects with at least four complete robot/tactile episodes.
The exact v3 implementation was byte-bound before those numeric payloads were
opened.

For every object, the highest carrier-complete episode ID was selected as target
from metadata and filenames. All other complete episodes of the same object were
used for source fitting. The target robot trajectory was a known intervention
input. Target tactile values were opened only after the source fit was content
bound.

## Interpretation

This is now promising evidence for the narrower robotics claim:

> Source-fitted Bayesian action-conditioned dynamics improve 32-frame real
> tactile-response forecasting over persistence, and the benefit degrades when
> the future action is permuted.

It does not yet establish dense 4-D geometric forecasting, unseen-object
zero-shot transfer, strict individual counterfactuals, calibrated joint
uncertainty, deployment safety, or state of the art.

## Remaining negative results

The covariance model is not ready for a probabilistic headline. In the reserved
confirmation, marginal 90% coverage was **75.0%**, joint coverage was **27.0%**,
and normalized joint ANEES was **115379**. The guarded deployment arm also
underperformed because two rejected objects fell back to a source-selected ridge
model rather than persistence. The paper should use the unguarded Bayesian action
ensemble for this result and treat guard/covariance repair as separate future
work unless independently revalidated.

## Provenance

- Development run: `33330455808`, artifact `9737547119`, SHA-256
  `de41ab129d22a118a397e21b588cbfc1bc8a9792d4be3a9e504734d47f8c4167`.
- Reserved readiness run: `33330710592`, artifact `9737630207`, SHA-256
  `84e8b463f9d3092739227985319e7910e1aaff3a59998b4e366455b8a7213920`.
- Reserved confirmation run: `33331368970`, artifact `9737749188`, SHA-256
  `bb82b44f9935dc839bfa293d044634219bac7e96fb7050f76c7f482cd874a382`.

The implementation, frozen protocols, compact evidence and object-level table
are collected in pull request #815 for ordinary protected-branch validation.
