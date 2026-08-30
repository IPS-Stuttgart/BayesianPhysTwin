# Reward-aligned stochastic-execution Slingshot certificate v4

## Question

Can the frozen Bayesian policy-gain guard improve the expected native Slingshot
reward on fresh continuous public-simulator worlds when rare late contact
bifurcations are treated as part of the execution distribution rather than as
malformed data?

## Why this is a new estimand

V3 correctly stopped when one of 128 calibration worlds exceeded its 0.5 mm
full-state duplicate tolerance. The two independent rollouts still agreed in
native reward within the registered 0.001 margin. V4 does not weaken or
reinterpret that terminal v3 result. It instead asks a decision-aligned
question on new worlds: the statistical unit is one world, one sensor draw,
and one native process realization.

Each world still requires eight ordinary fresh-process action rollouts, exact
world and action identity, common-prefix agreement, fixed endpoints, and
duplicate-incumbent reward agreement within 0.001. Full-state duplicate
distance is reported as process variability but is not an admission test. The
incumbent reward is the mean of independent action slots 5 and 7. This avoids
letting either incumbent process draw alone determine the policy comparison.

## Frozen method

- The 147 opened worlds, 161-dimensional feature, seven-neighbor gain
  predictor, posterior-mean candidate policy, guard, conformal ranks,
  simultaneous-regret comparator, bootstrap, and source gates are unchanged.
- Calibration uses 128 fresh worlds; evaluation uses 288 further fresh worlds.
- World seeds are 262080 and 262081. Sensor seeds are 262082 and 262083.
- Every future action runs once in a fresh Python process. There are no retries,
  replacements, partial scores, or target data.
- Candidate actions are sealed before their corresponding future actions.
- Evaluation futures remain inaccessible until the frozen pre-future barrier
  reproduces and passes.
- Any missing or malformed process, failed reward-repeatability check, or
  incomplete denominator stops the run and is retained as a technical failure.

The prior v3 outcome is used only to qualify this revised execution estimand.
It is not rescored, repaired, or included in either v4 partition. This is a
public-simulator source study, not an official benchmark, physical-safety, or
state-determinism claim.
