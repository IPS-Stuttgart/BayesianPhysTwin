# Deform360 reusable-trust execution lock v1

This execution lock closes an ambiguity left by the fresh-panel and physical-fit
locks: sharing only spring parameters while rebuilding object topology per episode
does not constitute a reusable PhysTwin.

The primary arm therefore builds one canonical object graph from frame zero of
the lowest frozen fit episode (`episode_0001`) for each object. Object springs and
rest lengths remain immutable across the other five fit episodes and all four
held episodes. Each episode may update only its frame-zero state/readout
registration and its gripper attachment springs from the known action geometry.

The lock was written after archive download and automatic undistortion, and after
robot-trajectory recovery had started, but before any fresh frame-zero object
reconstruction, mask/geometry inspection, fit outcome, or held outcome. This is
an implementation clarification, not a data-selected method change.

The episode-specific graph remains a fit-side control. It may demonstrate the
value of per-episode reconstruction, but it cannot fill Deform360's missing
reusable-PhysTwin benchmark cell. Held predictions still require all twelve
prediction hashes before any held outcome can be opened.
