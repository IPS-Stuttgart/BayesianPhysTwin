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

The graph candidate associates observed points to the current simulated mesh
using same-frame geometry. Assignment-mixture spread and metric sensor noise
enter the observation covariance. A shared sensor component is retained and is
not divided by point count. The innovation is robustified once. A candidate is
admitted only if it improves the disjoint prefix-validation interval under a
source-calibrated regret rule; otherwise the output is byte-identical to the
physical baseline.

The correction remains an observation/readout belief. It is not called a
physical state correction, and safety or contact claims may not be based on
corrected readout coordinates alone.

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

The primary metric is future symmetric L1 Chamfer distance in metres, matching
the benchmark's geometry convention. Secondary reports include Hausdorff
distance, early/middle/late Chamfer, predictive coverage, interval width,
energy score, and correction energy. The trial is the replication unit.

A positive result would establish a guarded online sim-to-real update on this
benchmark. It would not by itself establish better open-loop dynamics,
interpretable material parameters, or iid point-level coverage.
