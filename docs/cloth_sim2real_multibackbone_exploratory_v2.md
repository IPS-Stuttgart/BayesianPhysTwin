# Cloth Sim2Real multi-backbone exploratory study v2

## Question

The frozen Cloth Sim2Real v1 result showed that a guarded Bayesian readout
update improves dynamic MuJoCo continuation by 7.47% on an independent target
repeat. The original benchmark reports a stronger open-loop SOFA baseline for
some cloths. This study asks whether the same frozen update also improves a
stronger physical prior:

```text
SOFA physical rollout
        +
frozen v1 prefix-conditioned guarded readout update
```

The scientific test is compositional. No new readout candidate, graph rank,
prefix boundary, or admission threshold may be selected from this study.

## Evidence boundary

All three released real repeats were already opened by the v1
source/calibration/target sequence. Every v2 comparison on those repeats is
therefore **exploratory**. It can establish implementation feasibility and
estimate headroom, but it cannot provide a new independent confirmation or an
identical-information state-of-the-art claim.

Physical baseline generation:

- uses official benchmark commit
  `178a9b9722191c51cf0dcbc3cf0dc03701b09eb3`;
- reads the released simulator parameters and prescribed gripper trajectory;
- preserves the benchmark's backend-specific settling contract (ten seconds
  for SOFA and one second for MuJoCo);
- reads no real point cloud, prefix observation, or future outcome;
- records the simulator and runtime provenance in a sidecar;
- preserves `mujoco3` as the default and makes SOFA opt-in.

The readout update may use only the same allowed real prefix as v1. Future
point clouds remain inaccessible until a prediction seal exists.

## Frozen exploratory ladder

1. Reproduce the v1 MuJoCo baseline with the new adapter.
2. Run one SOFA dynamic smoke case (`chequered_rag_0/dynamic`).
3. Compare physical-only and guarded SOFA on the already-open repeat.
4. Continue to all three dynamic cloths only if the smoke produces finite,
   topology-consistent trajectories and the physical metric is in the
   published benchmark regime.
5. Report both the benchmark's directed L1 Chamfer metric and the v1 symmetric
   L1 Chamfer metric.

No quasi-static method adjustment is permitted. The v1 quasi-static
regression remains a negative transfer result.

## Advancement rule

A fresh preregistered evaluation is justified only if:

- guarded SOFA improves over physical SOFA on all three dynamic cloths;
- the object-balanced gain is positive for both directed and symmetric
  Chamfer;
- late-horizon behavior does not regress;
- the correction remains prefix-only and rejected cases are exact physical
  fallbacks;
- the result does not depend on retuning the v1 update.

A confirmatory claim then requires unused real executions, a new cloth
benchmark, or prospective data collection. The opened v1 repeats cannot be
relabelled as confirmation.

## Claim boundary

Even a positive result is an online continuation result because it observes a
real prefix. It may show that a guarded Bayesian state/readout belief update
improves a strong simulator prior, but it is not directly comparable to an
open-loop simulator that receives no real prefix.
