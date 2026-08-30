# DLO-Lab unknotting headroom development v1

## Purpose

This is a bounded development screen for a larger public-data contribution. It asks
whether the released DLO-Lab unknotting simulator contains enough
query-conditional action value to justify a later, separately frozen competence
predictor. It is not scientific evidence, a target evaluation, or an official
DLO-Lab leaderboard result.

The task is untouched by the earlier wrapping, slingshot, coiling, and separation
screens. The study uses only the public DLO-Lab simulator and its released knot.
It reads no held-v8, DLO4, DLO5, protected target, or newly recorded data.

## Frozen worlds and actions

Nine worlds rotate the released knot about its centroid by fixed angles from -35
to +35 degrees. Rotation preserves intrinsic geometry and the native unknotting
reward. The action bank is derived only from the public line joining native
control nodes 2 and 37:

- one shared-prefix hold;
- eight symmetric two-gripper pulls at fixed angular offsets around that line;
- two exact duplicate controls for deterministic replay checks.

Every action first executes the same two-macro, 30 mm vertical prefix. It then
holds or executes four 25 mm symmetric pulls. All worlds and actions are fixed
before native outcomes are generated.

## Native qualification

Each world must pass before another world starts. Qualification re-derives the
unchanged native final reward from sealed rope coordinates and checks finite
state, common-prefix identity, duplicate replay, world rotation, segment length,
rod height, and material attachment.

Material attachment is measured as drift from the initial gripper-to-control-node
offset. Absolute gripper-to-node distance is not used because the native
constraint may intentionally preserve a nonzero offset. This definition is
frozen before any unknotting outcome is observed.

If a world fails qualification, the study terminates without value analysis and
without retry. The incumbent fallback remains unchanged.

## Development gate

Only after all nine worlds are sealed and qualified may the compact reward bank be
opened. A useful query-conditional problem requires all of the following:

- the best fixed pull improves over shared-prefix hold by at least 0.05 reward;
- oracle-minus-best-fixed headroom, after a 0.002 numerical margin, is at least
  0.03 reward;
- at least three distinct actions are optimal across worlds;
- at least four worlds gain at least 0.03 over the best fixed action.

Passing would justify designing a separate off-grid source-transfer protocol. It
does not authorize that protocol automatically. Failing closes this exact action
bank without post-outcome tuning.

## Custody

The runner binds a clean Git revision, source file hashes, public source archive,
software runtime, protocol, output root, and an external write-once attempt
ledger. Preflight completes before the attempt is consumed. The registered run
is CPU-only, one-shot, and non-retriable.
