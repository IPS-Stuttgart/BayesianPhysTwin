# Deform360 dense reusable panel v1

This milestone freezes the prospective five-object prerequisite panel for a
reusable PhysTwin multi-episode result.

## Contact-conditioned amendment

The fixed `[110,191)` interval was rejected before any calibration dynamics or
target observation was opened. Across all 30 source actions it captured only
`25.09%` of the best equal-length controller displacement and `33.65%` of the
best path. The first replacement maximized known gripper displacement. It
failed a source-only smoke test on `002-rope-silk/0`: `83.07 mm` mean controller
motion coincided with only `0.597 mm` mean object motion because the selected
segment was gripper approach. The second prospective amendment selects an
81-frame window by path while closed, using only released robot actions and
aperture on the same six-frame grid. It does not inspect object motion, tactile
values, simulator output, or prediction metrics.

The action-aligned frame audit covers all 45 source/calibration action streams.
Calibration actions are QA metadata only; their object outcomes remain sealed.
All 15 protected target paths remain absent, and no target action, initial
frame, geometry, or tactile stream has been read.

- Canonical config hash: `1a78b8d74679ebf65768cc5078b34d034a2fcac55f7e0c0a00e50e1967a1c9bd`.
- Source action-window audit: `46eadfd9c94ca59a7dddd70c3a327f87604591aac3890404f0168ce535b1e4a6`.
- Action-aligned frame audit: `d4a3614535ec634d99993fd27f831eb915427e65f2342801c1b4557e4d85af2c`.
- Superseded 002 motion-floor diagnostic: `e10f0f5b43bc4d40e0eebe0655a745e7ba0dbde8c44058e3bafeab013fdef17a`.
- Displacement-only 002 source smoke: `5de2fa5dadace466a8c4e73538b2d55d29b18cf268a932219992e94d7b0d72e1`.
- Target-boundary audit: `dfec6a040c92a272ab987900f953adb26181e07348557a451136058bbd9f4bb7`.

## Decision

The amended panel is locked and contact-conditioned source admission may begin.
`081-stripe-rope` is a development object and is excluded. The fixed-window
`002-rope-silk` result is retained as a negative under-excitation diagnostic,
not as evidence against source-shared physics. The five targets remain sealed
until every source and calibration gate passes in the registered order.

This is a prerequisite panel. A state-of-the-art claim still requires the full
official Deform360 multi-episode split.

## Source graph-action-support discovery

Three source episodes were used to test an automatic frame-zero PhysTwin
candidate. Scarf episode 1 and squirrel episode 0 selected a graph-distance
action-support scale of `0.12 m` and an action-response gain of `0.9` on source
training frames only. Their untouched tails improved over persistence by
`23.26%` on the execution-balanced combined score. The selected values were
then frozen before evaluating rope episode 0, where the untouched-tail track
error fell from `35.95 mm` to `17.44 mm` and Chamfer distance fell from
`19.25 mm` to `8.18 mm`.

This is source-only discovery, not confirmation. The automatic graph is built
from each episode's frame-zero observations and is not yet one unchanged graph
reused across episodes. The current observation archive also lacks retained
contributor counts and calibrated metric covariance.

The resulting candidate is frozen by
`deform360_graph_action_support_independent_source_v1.json` with SHA-256
`d462fdfbaaccdce702292288976f71a7543174ffe1b8e6017550a019e315852b`.
The three examined episodes are excluded from confirmation. The next gate uses
the remaining 27 source episodes across all five objects. Calibration outcomes
and every target observation remain sealed.

Verification on `gpuserver6000` passed `560` tests with `1` intentional skip.
Focused Ruff lint and format checks passed. A fresh target-boundary audit found
all 15 protected derived-data paths absent.
