# Dynamic TAPNext++ Short-Sentinel V6 Source Result

## Decision

**The frozen V6 arm fails its source provider gate and is closed.** It does not
advance to hidden-future scoring, additional source cases, or fresh-object
evaluation.

No hidden identity, future object observation, target metric, V1 sealed case,
fresh-object artifact, or held-v8 artifact was read.

## Frozen Run

- Repository commit: `ca67db46ba6b23895a18cfa3125d018ce0a5e686`
- Opened source case: `059-shoe-ep0000`
- Tracker interval: frames 51--57
- Schedule digest:
  `8f85c28868955f5fcbb7cf4a9d117ca7ffad9c5ccf504c85fabf66aed0ee2964`
- Result digest:
  `35636b9c8ff1e765dbd174e33f25f16de72a7db91ee1f3fc853b3cf020ab7ea3`
- Report file SHA-256:
  `36a7c1cfdd31efe4ae4d50483b1dd9b5b6671bd9dd42155fde364dc743cdaea6`

The frozen target-free schedule filled all 9 active and 3 sentinel slots.
TAPNext++ processed seven frames in 2.49 seconds.

## Prefix Provider Outcome

Shortening the carrier improved accepted birth-and-update support from V5's
0/12 to V6's 3/12, but the preregistered requirements still failed:

- active identities supported at both endpoints: 3/9;
- sentinel identities supported at both endpoints: 0/3;
- all-sentinel requirement: failed;
- nine-inlier active pairwise gate: impossible;
- bias estimate: not formed;
- forecast update: not applied;
- post-prefix prediction: bit-exact persistence.

Role-specific inspection explains the support loss:

- at frame 51, 10/12 identities had a multiview proposal and 6/12 passed the
  three-view gate;
- two sentinels had no valid birth association;
- at frame 57, 5/12 identities had any proposal;
- only three active identities retained at least three independent views;
- no sentinel retained an endpoint proposal.

Because the candidate equals persistence after frame 57, it cannot pass the
locked accuracy gate. Hidden-future scoring was unnecessary and prohibited.

## Interpretation

V5 and V6 together show two distinct provider bottlenecks:

1. a long frame-zero carrier loses essentially every query;
2. a seven-frame carrier improves survival but physical projection visibility
   does not predict causal depth/mask association or tracker survival,
   especially for low-motion sentinels.

This closes physical-visibility-only selection for the fixed TAPNext++
sentinel interface. It does not justify lowering the three-view gate or
selecting a favorable source outcome.

A genuinely new source arm would need to condition the query schedule on
allowed prefix evidence before tracking, for example by requiring causal
depth/mask association support at both the birth and update frames. Such a
schedule must keep association probability separate from observation
reliability, robustify the state innovation only once, and preserve exact
fallback. It must be frozen as a new protocol before any hidden identity is
opened.

## Preserved Evidence

The checksummed report, query schedule, provider arrays, measurement archive,
and exact-fallback assimilation arrays are stored under:

`results/sota/diagnostics/deform360_dynamic_tapnextpp_short_sentinel_v6/059-shoe-ep0000/`
