# Retrospective DEFORM active-probe result

The content-bound result is
`results/development/deform_dlo45_active_decision_probe_v1/result.json`.
`validate_result.py` checks the selection, duration, action, harm, and pooled-RMSE
accounting before the result is accepted by CI.

## Pooled DLO4/DLO5 result

The evaluation contains all 28 official held trajectories and 532 fixed-horizon
decisions.

| Policy | Terminal RMSE [mm] | Gain vs fallback | Mean probe frames | Harmful decisions vs fallback |
| --- | ---: | ---: | ---: | ---: |
| Exact fallback | 58.164 | -- | 0.000 | 0/532 |
| Passive no-probe certificate | 55.673 | 4.28% | 0.000 | 0/532 |
| Active minimum-cost certificate | **54.106** | **6.98%** | **0.694** | 3/532 |
| Fixed 3-frame probe | 57.106 | 1.82% | 3.000 | 1/532 |
| Fixed 6-frame probe | 57.097 | 1.83% | 6.000 | 2/532 |
| Fixed 12-frame probe | 55.877 | 3.93% | 12.000 | 3/532 |
| Maximum outcome entropy | 42.846 | 26.34% | 6.823 | 42/532 |
| Hindsight probe/action oracle | 31.273 | 46.23% | 4.545 | 0/532 |

The active minimum-cost rule is 2.81% lower in RMSE than the passive certificate.
It also beats every fixed-duration certified policy while using 94.2% fewer
probe frames than the 12-frame policy. The maximum-entropy diagnostic is much
more accurate but has 42 harmful decisions and does not satisfy the registered
all-compatible-beliefs regret contract.

## Where active probing changes certifiability

- 71/532 decisions are certifiable without probing.
- The active portfolio certifies 119/532 decisions.
- Thus a positive-duration probe makes 48 additional decisions certifiable.
- In 34 of those 48 observed probe outcomes, more than one source hypothesis
  remains compatible. The active policy therefore frequently acts without
  identifying one complete source state.
- The selected positive-duration probes are distributed over 3, 6, and 12
  frames; most decisions either need no probe or fail closed to fallback.

The complete-trajectory direction is positive on both DLOs:

- DLO4: mean trajectory improvement 6.24%, bootstrap interval 2.66%--9.64%,
  14/0/0 wins/ties/losses;
- DLO5: mean trajectory improvement 7.45%, bootstrap interval 5.42%--9.61%,
  14/0/0 wins/ties/losses.

## Interpretation boundary

This is retrospective mechanism evidence, not a robot active-perception result.
The probe varies only the observed duration of the already recorded endpoint
motion. Internal-node probe response is taken from motion capture, not from a
learned visual provider. Alternative physical probe directions are not observed,
so the study does not establish counterfactual probe selection, unseen-object
generalization, target-domain regret control, deployment safety, or continuous
robot control.

The next decisive experiment must expose several physically executable probe
directions or amplitudes for the same initial condition and compare the exact
decision-value selector against state-entropy, query-variance, random, and no-probe
policies on a real terminal manipulation loss.
