# PokeFlex Bayesian Registration v1

## Purpose

The MatPhys LOO22 experiment showed that choosing more spring parameters cannot
close the published 8/15 mm gap: even the opened frozen-family oracle remains at
9.730 mm CD and 18.433 mm track error. The Deform360 prospective study supplied
the complementary diagnosis: geometrically coherent camera estimates can carry
large common-mode bias, especially with only the minimum number of views.

This protocol therefore tests a different Bayesian-PhysTwin contribution:

> update a persistent template state from reliability-aware real depth evidence,
> retain the physical prior where the observation is unsupported, and perform a
> causal one-step prediction with an exact fallback.

It does not modify the frozen Causal4D claim or the earlier PokeFlex Bunny split.

## Published reference

PokeFlex Table III reports the real-Kinect model at 6.498 mm unidirectional L1
Chamfer and 0.820 volumetric Jaccard. The value 5.00 in that row is normalized
ROI loss multiplied by 1000, not a 5.00 mm Chamfer score. The open evaluator at
commit `aaa8726` also contains an accumulator-label bug and refers to an older
internal validation naming scheme. We therefore register the released
checkpoint bytes but use a deterministic scorer implementing the paper's metric
definition.

## Causal boundary

For target frame `f`, every method receives only frames `f-5` through `f-1`.
Frame `f` depth, RGB, and mesh data are forbidden until scoring. Synthetic point
clouds sampled from target meshes are forbidden altogether. The target take's
template mesh is allowed because it is an explicit input to the published task.

The required comparison is:

1. template or exact persistence;
2. rigid ICP from the causal history;
3. the released Kinect checkpoint;
4. Bayesian graph registration plus an action-supported one-step prediction.

Observation reliability may use camera metadata, overlap disagreement,
visibility, source confidence, and robot/contact support. It may not use error
against the target or current physical state. The innovation enters once through
the robust likelihood. Two correlated Kinect views do not count as two
independent samples, and a common-mode camera-bias term remains explicit.

When support is inadequate, the candidate must return the physical/persistence
prior byte-for-byte. This is the main lesson transferred from Prob4D and the
prospective Deform360 failure analysis.

## Prospective split

`3dPrintedBunny` is excluded because its PokeFlex takes have already been used in
the earlier source-Warp study. The remaining 17 objects are split by metadata:

- development: five objects;
- covariance calibration: four objects;
- sealed target: eight objects.

Take `T2` is held out for every calibration and target object. `FoamDice_T3` is
the first development smoke because it is also the example named by the updated
official testing branch. Exact object lists and the information boundary live in
`configs/sota/pokeflex_bayesian_registration_v1.json`.

## Admission gates

A direct benchmark success requires mean target CD_UL1 below 6.498 mm and mean
Jaccard no worse than 0.820. The method must also improve CD_UL1 by at least 5%
over the released checkpoint, have a paired object-cluster bootstrap upper bound
below zero, and avoid more than 10% regression on any target object.

Calibration is fitted only on the four calibration objects. Target outcomes are
opened once after implementation, hyperparameters, calibration, scorer samples,
and checkpoint hashes are frozen.

## Development outcome

The first prospectively frozen action guard improved the released checkpoint by
2.41% on 20 previously unopened development takes but failed the locked 5%
transfer gate. Calibration and target objects remain sealed. See
`docs/pokeflex_action_guard_development_result.md` for the result and the
post-open common-mode-bias diagnostic.
