# Trajectory-conformal active decision sensing

This experiment wraps the frozen DEFORM DLO4/DLO5 decision-directed sensing policy from v3 with a trajectory-level split-conformal regret envelope.

## Question

Can an active physical-twin policy retain useful nonfallback decisions while extending a finite-support regret certificate to one exchangeable future complete trajectory?

The base policy is a deterministic mapping from the observed prefix and acquired internal-node measurements to further sensing, a terminal finite action, or fallback. It is fixed before calibration. For calibration trajectory `j`, define

```text
S_j = max over emitted nonfallback decisions d
      max(0, realized_normalized_regret_jd - finite_support_bound_jd).
```

The split-conformal radius is the `ceil((n+1)*(1-alpha))` order statistic of the complete-trajectory scores. On a new exchangeable trajectory, the event `S_new <= q` simultaneously bounds every nonfallback decision emitted by the same fixed policy on its realized sensing path. The operational wrapper retains the base action only when

```text
finite_support_bound + q <= registered_operational_tolerance;
```

otherwise it returns the exact base physical fallback. It never re-optimizes the terminal action after observing a probe outcome.

## Frozen primary point

- base policy: `decision_regret`;
- measurement budget: four internal nodes;
- calibration unit: complete trajectory;
- calibration trajectories: 18;
- disjoint source-test trajectories: 16;
- miscoverage: `0.20`;
- operational normalized-regret tolerance: `0.25`;
- bootstrap: 20,000 complete-trajectory resamples.

The primary tolerance is inherited from the pre-existing complete-plan controlled studies. The complete miscoverage/tolerance frontier is retained rather than selecting an operating point from source-test outcomes.

## Evidence boundary

The experiment uses only the nonoverlapping source replication inside the public DEFORM training partition. It opens no official evaluation file and collects no new data. The guarantee is trajectory-marginal under exchangeability and applies to the registered loss and one fixed active policy. It is not a pointwise conditional, unseen-object, learned-sensor, online-robot, arbitrary-action safety, or deployment guarantee.

The workflow reproduces the predecessor v3 result at its exact source revision before applying the conformal wrapper, then binds all outputs by content hash.
