# Deform360 Released-Particle Warp Readout Source V1

Status: locked before any matched-origin dense source score was computed.

This development-only audit asks whether the already admitted official
PhysTwin/Warp rope backend transfers from the old 21-node source evaluator to
the ordered particles subsequently released by the Deform360 authors. It uses
only the previously opened `001-rope` source episodes 0, 3, 4, 5, and 8.
Episodes 1, 2, 6, 7, and 9 remain forbidden.

The executable lock is
`configs/causal4d_public/deform360_released_warp_readout_source_v1.json`.
It binds the old source observations, author-released metadata, split and robot
files, the exact released particle frames, the previously selected physical
candidates, matched rollout origins, metrics, and the transfer gate.

## Question

The prior source gate established that a sparse official-Warp forward model
could beat sparse persistence on three of five leave-one-source folds. It did
not establish that the same physical predictions improve the dense,
fixed-identity particles used by the released Deform360 evaluator.

This audit tests that missing interface:

1. restart Warp from the latest causal 15 Hz state before each released test
   window;
2. use the source-gate candidate selected without that episode's sparse error;
3. attach every released origin particle to the nearest point on the 21-node
   rope polyline;
4. transport the fixed local offset with the predicted segment orientation;
5. compare the dense forecast with exact matched-origin persistence.

## Fair Origin

The archived source-gate trajectories begin near action onset, whereas the
released persistence baseline begins immediately before each test window.
Comparing those trajectories directly would give the methods unequal forecast
horizons. The primary arm therefore reruns Warp from the same last available
15 Hz pre-test state used by persistence.

Initial velocity is estimated causally from the two last sparse states. A
zero-velocity rerun is a declared control. No future object particle affects
the simulator, association, candidate identity, or readout rule.

## Dense Readout

At the matched origin, each ordered released particle is projected onto the
closest clamped segment of the sparse polyline. Segment identity, barycentric
coordinate, and local offset are then frozen.

For the primary readout, the local offset is rotated by the minimum rotation
from the origin segment tangent to the predicted segment tangent. Antiparallel
and degenerate segments have deterministic rules in the lock. A fixed-offset
translation-only readout is retained as a diagnostic. Future particles are
used only after prediction sealing to compute metrics.

This attachment yields direct readout support for every released particle, but
it does not turn the 21-node model into a dense reconstructed PhysTwin. It is a
fixed material readout of a sparse official-Warp state.

## Arms And Gate

The primary arm uses the locked leave-one-source candidate, causal
finite-difference velocity, and rotated offsets. Controls are matched
persistence, zero initial velocity, pooled candidate 115, fixed-offset
transport, and the unmatched archived long rollout.

The route passes only if the primary arm:

- improves mean dense Chamfer by at least 5% over matched persistence;
- wins Chamfer on at least three of five episodes;
- does not worsen panel mean ordered-particle identity error;
- remains finite and below 50% p99 sparse-edge strain.

A pass justifies a new, separately preregistered evaluation on fresh objects.
A failure stops this readout route.

## Claim Boundary

All five episodes were already open, so this is development evidence, not
independent confirmation. The author release does not include enough evaluator
detail to authorize a direct comparison with Deform360 Table 4. This audit
therefore cannot establish state of the art, calibrated uncertainty, or a
dense Bayesian-PhysTwin result.

No held-v8 artifact, sealed PokeFlex target, forbidden rope episode, or
Deform360 target split is accessed by this protocol.
