# Cloth Sim2Real online-belief protocol

## Purpose

Deform360 has no admissible fresh public object left after taking the
conservative union of all existing source, calibration, target, reserved, and
technical dispositions. V12, V13, and V14 remain frozen negative results and
must not be revived with a recovered exclusion manifest.

This protocol moves the guarded Bayesian-PhysTwin update to the independent
public Cloth Sim2Real benchmark. The benchmark supplies three cloth materials,
three trials per cloth, dynamic and quasi-static bimanual actions, real
point-cloud trajectories, released simulator code, and published MuJoCo,
Bullet, Flex, and SOFA comparisons.

The scientific question is:

> Can a causal real-observation prefix update a strong cloth-simulator prior
> while retaining an exact physical fallback, and improve the untouched future
> on an independently repeated trial?

This is an online continuation task. It is not presented as an open-loop
comparison with identical information to the original benchmark.

## Frozen split

The split was fixed from filenames before any point coordinate or metric was
read:

| Repeat suffix | Role | Uses |
| --- | --- | --- |
| `_0` | Source | Method and hyperparameter selection |
| `_1` | Calibration | Frozen transfer gate and uncertainty calibration |
| `_2` | Target | Prefix-only prediction, then sealed future evaluation |

Each role contains the three cloths and both tasks, yielding six cases per
role. Dynamic manipulation is primary because it has the largest published
sim-to-real gap. Quasi-static manipulation is secondary and tests contact
transfer.

Within every case, the branch is fixed at 25% of the available frames. The
first 60% of that prefix fits the update. The remaining prefix frames are a
disjoint causal validation interval. The future begins strictly after the
branch. The outcome-blind inventory found 130 frames per dynamic trial
(`fit_stop=19`, `branch=32`) and 420 frames per quasi-static trial
(`fit_stop=63`, `branch=105`).

The generated dataset manifest has artifact SHA-256
`721d85ee4411901d88917640b987e9baf125ec9776c3ac901fec32dd4660a433`
and file SHA-256
`18f23c9a508939f20502a960130f5de00ce0161ef037592933c0ec3ac0ce58e0`.
It records only directory names, frame names, frame counts, and byte counts;
no point coordinate or metric was read while constructing it.

## Baseline and candidate

The unchanged baseline is the released MuJoCo cloth implementation with the
released cloth/task parameters and Cartesian action trajectory. No parameter
may be refit on calibration or target repeats.

The candidate ladder is:

1. unchanged physical baseline;
2. prefix-estimated global translation at readout;
3. graph-smoothed prefix discrepancy at readout;
4. bias-aware, baseline-relative guarded graph discrepancy.

The graph candidate associates each physical vertex with a four-candidate
local mixture in the same-frame observed cloud. Candidate geometry determines
the association distribution, but it does not determine prior perception
reliability. Assignment-mixture spread and metric sensor noise enter the
observation covariance. A 5 mm shared sensor component is retained and is not
divided by the number of dense points or duplicate rows. The state innovation
is processed by the existing Gaussian/broad-Gaussian robust mixture and is not
also reused as prior reliability.

Candidate selection uses only the disjoint prefix-validation interval. An arm
must improve its mean symmetric L1 Chamfer by at least 2%, win at least 60% of
validation frames, and not worsen the worst validation frame. Otherwise the
output is exactly the physical baseline.

The correction remains an observation/readout belief. It is not called a
physical state correction, and safety or contact claims may not be based on
corrected readout coordinates alone.

## Development smoke and method lock

`chequered_rag_0/dynamic` was declared as the source development smoke. The
production runner independently reproduced the released MuJoCo rollout
bit-for-bit. It then selected `graph_l1_s1` from the finite locked bank using
frames 19--32, sealed frames 33--129, and only afterward opened that source
future.

The selected arm improved held-out prefix Chamfer by 35.90% and untouched
future symmetric L1 Chamfer by 9.58%. The directed L1 metric used by the
published benchmark improved by 9.05%, and symmetric Hausdorff distance
improved by 11.59%. Raw nominal 90% coordinate coverage was only 56.43%, so
the smoke supports the mean update but explicitly does not establish
calibration.

The method and finite candidate bank are frozen in
`configs/sota/cloth_sim2real_online_belief_v1_method_lock.json` before the
remaining five source cases are read. No calibration or target point cloud was
opened.

## Gates

Source advancement requires at least 5% mean dynamic future-Chamfer
improvement and joint non-regression on all three source cloth trials.

Calibration advancement requires at least 5% mean dynamic improvement, at
least two of three cloth wins, no cloth worse by more than 5%, and frozen
calibration before any target prefix is read.

Only if both gates pass may target prefixes be processed. Every target
prediction is sealed before its future point clouds are made available to the
evaluator.

## Metrics and claims

The protocol primary metric is the mean of the two directed nearest-neighbour
L1 distances. The published benchmark's simulator-to-observation directed L1
distance is reported alongside it for direct comparison. Secondary reports
include Hausdorff distance, early/middle/late Chamfer, predictive coverage,
interval width, energy score, and correction energy. The trial is the
replication unit.

A positive result would establish a guarded online sim-to-real update on this
benchmark. It would not by itself establish better open-loop dynamics,
interpretable material parameters, or iid point-level coverage.
