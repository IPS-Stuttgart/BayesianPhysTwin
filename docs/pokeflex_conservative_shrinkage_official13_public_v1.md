# PokeFlex public official-subset evaluation v1

## Purpose

The exact official-18 evaluation is blocked because five evaluator IDs are absent from
the public PokeFlex archive and no authoritative public mapping exists. This protocol
keeps the official take identities that can be materialized exactly and asks a narrower
question:

> Does the unchanged, source-selected Bayesian-PhysTwin state update improve the released
> PokeFlex checkpoint on the publicly available subset of its official validation takes?

This is a paired checkpoint comparison. It is not a reconstruction of the paper's
18-object aggregate and cannot establish that the candidate is below the published
`6.498 mm` score.

## Frozen cohort

The cohort contains all 13 exact official validation take IDs found in the public archive.
Ten are prospectively untouched by this method. Three are disclosed development-overlap
controls: `FoamDice_T3`, `PlushOctopus_T6`, and `ToiletPaperRoll_T1`.

The five unavailable official IDs remain recorded in the protocol and receive no
replacement: `Pillow_T8`, `3dPrintedCylinder_T7`, `3dPrintedHeart_T14`, `Sponge_T10`, and
`3dPrintedPizza_T13`.

## Method and custody

The candidate remains
`checkpoint_action_local_state_relative_0.4_residual_scale_0.125`, selected once on the
nine-object source panel. It performs robust graph registration at frame `f-1`, transports
the correction one material-identity step, and applies scale `0.125` to the released
checkpoint prediction at frame `f`. There is no target-tuned selector.

Prediction at `f` may use Kinect and robot history only through `f-1`. Missing action-field
inputs or rejected updates return the released checkpoint byte for byte. All 13 prediction
archives must be sealed at one clean implementation revision before any campaign target
deformation mesh is opened. A failed take is retained; no replacement is allowed.

## Registered evaluation

The full 13-take equal-frame CD and object-balanced CD are descriptive. The primary gate is
computed only on the ten prospectively untouched takes. It requires all of the following:

1. Prospective object-balanced `CD_UL1` improves over the released checkpoint.
2. The 97.5% object-bootstrap upper bound on candidate-minus-baseline CD is below zero.
3. No prospective object regresses; an exact byte-identical fallback may tie.

The three overlap controls cannot cause the gate to pass or fail. The published `6.498 mm`
and `0.820` Jaccard results are contextual only. Boolean-volume Jaccard is reported when it
is defined on the released meshes, never repaired or replaced, and is non-gating.

## Interpretation

If the gate passes, the supported claim is that the frozen Bayesian state update improves
the released PokeFlex checkpoint on ten prospectively untouched takes from the public
official-validation subset, with all 13 public official takes reported. It is not a claim
about the unavailable five takes or the paper's complete aggregate.

If the gate fails, this correction is closed as a PokeFlex SOTA route under the registered
method. The result must still be archived with technical failures, fallbacks, overlap
controls, and invalid Jaccard rows intact.

The canonical protocol is
`configs/sota/pokeflex_conservative_shrinkage_official13_public_v1.json`, SHA-256
`6becd1a69d482654bf4e4faad57cc94c76ab56c4e1d24e46707e7a423aff0146`.
