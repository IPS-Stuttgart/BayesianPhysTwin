# PokeFlex public paired evaluation v1

## Question

Does the frozen Bayesian-PhysTwin state-shrinkage arm improve the released
PokeFlex Kinect checkpoint on public recordings that were absent from all
tracked Bayesian-PhysTwin development history at lock time?

This is the strongest public prospective comparison available after the exact
18 internal validation identifiers in the upstream evaluator proved not to
have a complete, documented mapping to the released public archives.

## Frozen cohort

The selection audit scanned all 192 local and origin branch tips and all 3,826
unique tracked text blobs. For every eligible object, it selected one public
take that did not occur in that history using the minimum salted SHA-256 rule
recorded in
`configs/sota/pokeflex_public_paired_selection_v1.json`. The resulting cohort
contains 15 takes from 15 distinct physical objects. Replacement is forbidden.

Freshness is repository-relative: it means the takes were absent from the
complete tracked Bayesian-PhysTwin history at lock time, not that no external
researcher has ever examined these public recordings.

## Frozen method

The candidate is unchanged from the passed source gate:

`checkpoint_action_local_state_relative_0.4_residual_scale_0.125`.

For target frame `f`, the method uses Kinect and robot history only through
`f-1`. A frame without an accepted registration, active-force support, and the
complete registered robot-pose history is unsupported and returns the released
checkpoint prediction byte-for-byte.

## Custody

1. Lock the protocol, cohort, implementation, source result, upstream commit,
   and checkpoint hashes.
2. Produce all 15 target-free prediction seals at one clean implementation
   revision.
3. Build and validate the complete prediction barrier.
4. Only then open the selected future meshes and score once.

No selected take may be replaced, and no outcome may alter the method or gates.

## Primary gates

The paired object-balanced comparison passes only when all conditions hold:

- candidate relative `CD_UL1` improvement is positive;
- the 97.5% object bootstrap upper bound on candidate-minus-baseline error is
  below zero;
- no object regresses;
- at least 12 of 15 objects improve;
- at least 12 of 15 objects contain at least one supported update.

The published PokeFlex Kinect value of 6.498 mm is reported only as cross-split
context. Jaccard is non-gating and uses the public evaluator's default Trimesh
reconstruction semantics; invalid booleans remain reported as invalid.

## Claim boundary

A pass supports the statement that the frozen Bayesian state update strictly
improves the released PokeFlex checkpoint on the registered 15-object public
cohort. It does not reproduce the unavailable internal validation split and is
not, by itself, an exact published-table SOTA comparison.
