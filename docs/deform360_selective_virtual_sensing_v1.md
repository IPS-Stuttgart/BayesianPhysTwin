# Deform360 Selective Full-Field Virtual Sensing V1

Status: locked before download or media access for the 12 selected objects.

The executable lock is
`configs/sota/deform360_selective_virtual_sensing_v1.json`. Its canonical
checksum is
`af4100e373233280a1c39e35fe38d213e611b8c8ff852307c151f148c43fbb87`.
The lock is validated by
`bayesian_phystwin.deform360_selective_virtual_sensing_protocol`.

## Paper hypothesis

A small number of causal multiview camera tracks can update a dense deformable
state without training on the target object. The update should help points that
were never observed, persist into future frames after each observation, reject
identity-inconsistent tracks, and return the base trajectory exactly when the
measurement is not trustworthy.

The intended paper claim is:

> A training-free, correspondence-selective recursive discrepancy field turns
> sparse camera prefixes into improved hidden full-field forecasts across new
> filament, sheet, and volumetric objects.

This is a new online-assimilation setting, not an official Deform360 Table 4
comparison. Virtual sensing, RBF interpolation, Bayesian filtering, and
conformal risk control are established ideas individually. The contribution is
their causal, simulator-agnostic composition with raw multiview tracking,
hidden-point evaluation, future-only scoring, and exact fallback for deformable
digital twins.

## Development evidence

The method was developed on 27 already-open Deform360 episodes from five
physical objects. Sixteen frame-zero centers were tracked from eight cameras at
frames 19, 38, and 57. Every center was permanently removed from the score;
only frames after an update were scored.

The selected physical/persistence pairwise-gated arm improved over the sealed
physical prior by 28.83% in hidden identity RMSE and 24.15% in hidden symmetric
Chamfer. It won 24/27 and 23/27 episodes, respectively. Against the selected
raw backbone, it improved by 15.74% and 14.10%; all five object means improved,
and both object-cluster intervals excluded zero.

The simpler persistence-only pairwise arm reached 7.646 mm hidden identity
RMSE and 6.787 mm hidden Chamfer, versus 9.357 mm and 8.243 mm for persistence.
That is an 18.29% and 17.66% reduction and improved all five development
objects. It is the prospective primary arm because it requires no target-fitted
physical model. The physical/persistence selector remains a secondary
composition experiment wherever a physical prediction was sealed without
target outcomes.

These numbers selected the method and are not confirmatory evidence. The
archived development summary is under
`results/sota/diagnostics/deform360_raw_alltracker_pairwise_gate_v1/`.

## Frozen method

For each 76-frame episode:

1. Reconstruct frame-zero material points without reading later dense states.
2. Select 16 deterministic farthest-point centers visible in at least two
   cameras.
3. At updates 19, 38, and 57, run the pinned AllTracker model on exactly RGB
   frames `[0,u]` in eight cameras and robustly triangulate the centers.
4. Compare pairwise center distances before and after observation. Accept the
   largest deterministic consensus clique only if it has at least 9 centers and
   70% of available support under the frozen 30 mm plus 10% strain tolerance.
5. Update the frozen global plus local Student-t RBF discrepancy state on the
   accepted clique and add its decayed field to persistence.
6. On insufficient support or correspondence rejection, emit persistence
   bit-for-bit.

The belief parameters, tracker revision and checkpoint, camera count, center
count, update frames, correction cap, and correspondence thresholds are all in
the lock. Held outcomes may not change them.

## Prospective cohort

Only top-level directory names from the pinned Hugging Face snapshot were used
to select the cohort. None of these object paths existed on either compute
server before the lock.

| Stratum | Objects | Episodes |
| --- | --- | --- |
| Filament | thread; jump rope; climbing rope; hemp rope | 5/2; 1/4; 0/7; 8/0 |
| Sheet | yellow cloth; handkerchief; wipe cloth; wrap paper | 9/3; 0/4; 8/5; 2/8 |
| Volumetric | doll; ball; rabbit; frog | 0/1; 5/2; 8/9; 5/2 |

The two episodes per object are the first two SHA-256 ranks among IDs 0 through
9 under the fixed seed `deform360-selective-virtual-sensing-v1`. Successful
episodes are never selected or discarded using outcomes. Failed objects are
not replaced.

The six objects reserved for the earlier frame-zero-only experiment are not
used. Their seal therefore remains intact.

## Information order

1. Commit and publish the code, tests, method lock, and source hashes.
2. Download only the 12 named objects at the pinned dataset revision.
3. Build frame-zero geometry, calibration assets, and causal RGB measurements.
4. Write and hash every measurement and prediction artifact.
5. Verify that the builders did not read future dense reconstructions, particle
   tracks, or target metrics.
6. Only then construct or open full future targets once for scoring.
7. Report all successful episodes, all object means, all strata, all failures,
   and every predefined comparator.

Any future dense reconstruction, particle track, or target metric opened before
the corresponding prediction hash invalidates that object without replacement.

## Confirmation gate

Episodes are averaged within object and objects receive equal weight. Both
co-primary metrics must pass every gate:

- at least 9 evaluable objects and at least 3 in every stratum;
- at least 10% object-balanced improvement over persistence;
- an object-clustered 95% interval whose upper endpoint is below zero;
- one-sided exact object-level sign-test `p <= 0.05`;
- no stratum mean regression;
- no object regression larger than 10%.

The conjunction is an intersection-union claim, so success requires both hidden
identity RMSE and hidden symmetric Chamfer. A failed gate is a prospective
negative or mixed result. It cannot trigger a replacement arm, threshold, or
cohort.

## Paper threshold

Passing the locked gate would be worthy of a strong paper when accompanied by:

- the selected physical-prior composition on the existing development panel;
- persistence, raw-backbone, ungated-RBF, CPD, and corruption controls;
- risk-coverage and exact-fallback plots;
- qualitative hidden-point trajectories and failure cases;
- runtime and camera/center-count ablations fixed on development data.

It would establish prospective cross-object virtual sensing, not direct
Deform360 leaderboard superiority. Official split and evaluator parity remain a
separate requirement for any state-of-the-art wording.
