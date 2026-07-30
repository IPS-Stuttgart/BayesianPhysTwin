# Prospective RGBench Online-Belief Protocol

## Purpose

This study tests whether a guarded Bayesian readout update can turn a
reproducible public cloth simulator into a stronger causal continuation model
on RGBench. It is separate from the frozen Causal4D and Deform360 studies.

The benchmark is RGBench at upstream commit
`eddae2f28f388b4706d65d626f67bc9e34b14c68`, with public dataset revision
`136c00dc5f96b6b3d20427e93875a1c00d7a7cc9`. The immutable metadata-only
dataset lock has canonical digest
`3789947cbb9c7c58ccc39b8186b4e30e2a258e615bc2e02c05842bcdafe160e8`.

## Prospective split

Only the seven garments with published RGBench baselines and samples
`01`--`03` enter the primary study. This exactly matches the published
three-sample cell size. The two released non-manifold garments are excluded
before outcomes because the paper has no baseline for them.

A fixed salted SHA-256 order assigns whole garments:

- source: `white_cakeskirt`, `brown_coat`, `green_tshirt`;
- calibration: `grey_pleat_skirt`, `white_shirt`;
- target: `blue_dress`, `beige_hoodie`.

This yields 27 source, 18 calibration, and 18 target captures. Generalization
is across unseen garments, not merely held-out repetitions of the same
garment.

The lock was produced from point-cloud filenames and byte sizes, robot
trajectory timestamps, required stream checks, calibration hashes, and mesh
topology counts. It did not parse point coordinates or compute an outcome
metric. The shortest published cell has six evaluation frames, so the frozen
minimum information budget is two fit frames, one disjoint validation frame,
and three untouched future frames. A short prefix does not bypass the
admission gate.

## Method

The physical baseline is the official RGBench PyBullet fixed-point wrapper.
Simulation may use the complete released actuator trajectory because the
future action is known, but it may use only point-cloud filenames before a
prediction is sealed.

Within the first 25% of the evaluation sequence, the method:

1. associates simulator vertices with the permitted real point clouds;
2. includes assignment-mixture spread and a shared-bias floor in metric
   observation covariance;
3. applies the existing robust mixture likelihood once to innovations;
4. estimates global and graph-smoothed readout corrections;
5. selects an arm on a disjoint prefix-validation block using RGBench's
   primary real-to-simulation L1 Chamfer;
6. admits the correction only if it improves validation by at least 2%, wins
   at least 60% of validation frames, and does not worsen the worst frame;
7. otherwise returns the PyBullet rollout bit-for-bit.

The candidate equals PyBullet throughout the observed prefix. An admitted
correction is applied only after the branch point. Thus the full-window score
does not reward fitting the prefix itself.

## Metrics and gates

The primary score is the RGBench paper metric: mean real-to-simulation
Manhattan Chamfer over the full published evaluation window. Future-only
primary error, sim-to-real Chamfer, symmetric Chamfer, Hausdorff distance,
early/middle/late future errors, coverage, interval width, and energy score
are secondary diagnostics.

Source opens calibration only if all of the following hold:

- object/action-balanced improvement over PyBullet is at least 5%;
- all three source-garment means are non-regressing;
- at least six of nine garment/action cells improve;
- the aggregate candidate is below published GarmentDynamics;
- at least six of nine cells beat published GarmentDynamics.

Calibration opens target only if:

- object/action-balanced improvement is at least 3%;
- both calibration-garment means are non-regressing;
- at least four of six cells improve;
- the aggregate candidate is below published GarmentDynamics;
- at least four of six cells beat published GarmentDynamics.

The 18 calibration captures provide the trial-level order statistic used for
the nominal 90% uncertainty scale. Rank 18 of 18 is used. The resulting claim
is marginal trial-level coverage under exchangeability, not simultaneous
coordinate, horizon, action, or garment coverage.

## Claim boundary

This is a prospective comparison against published SOTA numbers, but it is not
an identical-information open-loop simulator comparison. The candidate sees
an early real point-cloud prefix while GarmentDynamics does not. A positive
result supports the narrower claim that guarded online Bayesian correction
can outperform the published open-loop simulator after observing a causal
response prefix. It does not establish a universally better cloth simulator
or identify the correction as a material-state update.
