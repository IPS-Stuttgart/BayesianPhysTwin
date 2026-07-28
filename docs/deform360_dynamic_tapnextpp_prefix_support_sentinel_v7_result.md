# Dynamic TAPNext++ Prefix-Support Sentinel V7 Source Result

## Decision

**The frozen V7 arm fails its source provider gate and is closed.** No hidden
future identity or target metric was read. It does not advance to additional
source cases or fresh-object evaluation.

The V1 sealed cohort, all fresh-object artifacts, and held-v8 remain untouched.

## Frozen Run

- Repository commit: `83e8799fc56af966727afca5e583d967e818af13`
- Opened source case: `059-shoe-ep0000`
- Prefix-support screen digest:
  `2fea2d1cc6b97b74f5cedea78304587c3dd131348de2271e4d87866af5b1b65a`
- Query schedule digest:
  `2c73a37043217c29701eddddd868dd4d5f749848b00b6f509925c5a72df92088`
- Result digest:
  `0916c408c74463c3c2e99eb477233a4407898917fcc95c4278c76998d2c1a22e`
- Report file SHA-256:
  `2716183931562b078e42aff08a66a7bf94e00aa6fe4b32513948f33e6c025e0c`

## Prefix Screen Outcome

The V7 screen succeeded:

- graph nodes screened: 452;
- nodes with at least three depth/mask association candidates at both frames
  51 and 57: 265;
- full 9-active/3-sentinel physical schedule: available and sealed.

This establishes that the source prefix contains abundant multiview geometric
support. V6's poor support was not caused by a lack of observable depth/mask
surface near the physical nodes.

## Tracker Outcome

TAPNext++ still failed the registered support gates:

- active identities supported at both endpoints: 4/9;
- sentinel identities supported at both endpoints: 0/3;
- all-sentinel requirement: failed;
- nine-inlier pairwise gate: impossible;
- common-bias estimate: not formed;
- state update: not applied;
- post-prefix forecast: bit-exact persistence.

The screen improved active support from V6's 3/9 to 4/9 but did not make any
low-motion sentinel trackable through the seven-frame interval. Since the
candidate equals persistence, it cannot pass the locked hidden-accuracy gate,
and hidden scoring was neither needed nor permitted.

## Interpretation

The V5--V7 ladder now localizes the fixed interface failure:

1. physical projection provides abundant potential identities;
2. causal depth/mask association exists at both endpoints for most nodes;
3. TAPNext++ does not preserve enough of those material identities, especially
   low-motion sentinels, under strict multiview fusion.

This closes further camera-tracker query-scheduling work on this opened case.
Lowering the three-view gate, changing the query interval again, or selecting
identities using this tracker outcome would be post-open tuning.

The next credible source arm should remove the failed carrier rather than tune
it: use the already-supported multiview depth associations at frames 51 and 57
as direct sparse pseudo-measurements. The same sentinel common-bias estimator,
metric covariance propagation, robust innovation, disjoint hidden scoring, and
bit-exact persistence fallback must remain. This would test an independent
depth endpoint channel rather than another RGB tracking heuristic.

## Preserved Evidence

The screen, schedule, provider, measurement, assimilation, and report artifacts
are stored under:

`results/sota/diagnostics/deform360_dynamic_tapnextpp_prefix_support_sentinel_v7/059-shoe-ep0000/`
