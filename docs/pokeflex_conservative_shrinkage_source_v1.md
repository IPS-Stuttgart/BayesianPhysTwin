# PokeFlex Conservative Shrinkage Source v1

## Purpose

Earlier PokeFlex experiments established two useful limits. Independent D405
depth predicts candidate regret, but a learned source-object guard failed on
new objects. Strong camera-derived state corrections also have enough capacity
to improve several objects while producing a large tail regression on the
3D-printed pyramid.

This source-only protocol tests the simpler hypothesis that the transferable
component is a deliberately weak action-local correction. It uses no learned
online selector. Hidden errors select one fixed arm only on previously opened
source objects; the eight original target objects remain sealed.

## Selection Rule

The finite candidate bank is inherited unchanged from the parent PokeFlex
protocol. A positive-scale arm is eligible only if it:

1. improves the equal-object source mean by at least 1%;
2. does not regress any opened physical object; and
3. preserves exact checkpoint fallback on every frame lacking both an accepted
   graph update and action support.

Among eligible arms, selection is lexicographic: smallest positive scale,
smallest support radius, then arm name. This implements the parent protocol's
predeclared strongest-shrinkage preference and prevents the high-capacity arm
from winning merely because the vulnerable object is held out.

The rule is repeated in nine leave-one-object-out folds. Advancement requires
the same arm in every fold and improvement on every held object. These are
source-development checks, not confidence intervals or target evidence.

## Information Boundary

Every input artifact predicts frame `f` from Kinect frames `f-5` through
`f-1`, plus robot history through `f-1`. The candidate correction is estimated
at `f-1` and transferred once by fixed template-vertex identity. Frame `f`
observations never form the prediction. Source target meshes are used only to
choose the fixed family.

No target archive member may be inspected until a separate target protocol,
prediction/scoring split, code revision, checkpoint inventory, and eight-case
prediction barrier have been committed and pushed.

## Claim Boundary

A passing source gate only authorizes drafting the target protocol. It is not a
state-of-the-art result, a calibrated uncertainty claim, a material-parameter
result, or a Causal4D counterfactual result.
