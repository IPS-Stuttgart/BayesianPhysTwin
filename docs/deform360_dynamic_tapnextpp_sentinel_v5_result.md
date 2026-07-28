# Dynamic TAPNext++ Sentinel V5 Source Result

## Decision

**The frozen V5 arm fails its source gate and is closed.** It must not advance
to additional source cases or a fresh-object evaluation.

This is a provider-support failure, not a hidden-future accuracy result. No
hidden identity, future object observation, V1 sealed target, held-v8 artifact,
or target metric was read.

## Frozen Run

- Repository commit: `50e2c2c0e64e4c172c7534fb39fbaa0772b23618`
- Opened source case: `059-shoe-ep0000`
- Schedule digest:
  `36a2e07cef9e6ff5b5b82bc9cfa6bdfd07c3717978d515bac0160c3e782933dd`
- Result digest:
  `c8099e2b971f6376ae271afa2bbd7377f66f3ff89581e6a4f2f799779945ad8a`
- Report file SHA-256:
  `46c2a334ad5180523bce1c3c6357c22888002743e2266f15db60cb455f554e6c`

The target-free physical schedule successfully contained:

- 9 active identities with 15.35--46.85 mm predicted prefix motion;
- 3 sentinel identities with 0.405--0.499 mm predicted prefix motion;
- 7--8 predicted visible cameras for every identity.

The schedule hypothesis was therefore geometrically feasible before tracking.

## Prefix Provider Outcome

TAPNext++ produced a proposal for all 12 identities at frame 0. At frame 57:

- only 1/12 identities had any multiview proposal;
- that proposal had only two independent views;
- 0/12 identities passed the frozen three-view support gate;
- 0/9 active identities were available;
- 0/3 sentinel identities were available.

The sentinel common-bias estimator therefore returned
`incomplete-sentinel-endpoint-support`. The arm made no update and produced
bit-exact persistence for every forecast frame after frame 57.

Because the candidate equals persistence, it cannot improve over both
persistence and the selected backbone. The advancement gate fails without
opening hidden-future metrics.

## Interpretation

This closes a **single frame-zero-to-frame-57 sentinel carrier**. It does not
reject common-mode sentinel debiasing in general. The failure occurs before
bias estimation: the carrier is too long for this provider and multiview gate.

A future arm may use a newly preregistered short-horizon sentinel birth near
the update. The physical source rollout supports this possibility without
reading observations or outcomes: for births at frames 39, 45, and 51, the
frame-57 physical displacement has more than 300 active candidates above
2 mm and more than 120 sentinel candidates below 0.5 mm. Such an arm must use
a new protocol and frozen code; V5 thresholds or support rules must not be
relaxed after this result.

## Preserved Evidence

The checksummed report, query schedule, provider arrays, debiased measurement
archive, and assimilation arrays are stored under:

`results/sota/diagnostics/deform360_dynamic_tapnextpp_sentinel_v5/059-shoe-ep0000/`
