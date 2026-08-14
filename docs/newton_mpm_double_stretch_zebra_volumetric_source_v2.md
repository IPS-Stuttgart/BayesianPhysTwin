# Volumetric Newton MPM source gate v2

## Question

Can a mass-preserving volumetric Newton implicit-MPM backend with separate
benchmark-identity readout and compliant finite-mass contact avoid the failure
of the frozen direct-particle, hard-attachment v1 adapter?

This is a one-case development mechanism gate on the already-open public
`double_stretch_zebra` case. It is not a fresh evaluation, a PhysWorld
reproduction, a calibration claim, or evidence of state-of-the-art accuracy.
No held-v8, target, or confirmation artifact is in scope. Prob4D is unused and
MolmoMotion is fixed at `beta=0`.

## Distinct hypothesis

The v1 adapter treated every sparse, surface-heavy benchmark identity as an
MPM material particle and made attached particles kinematic. V2 changes only
the representation and contact interface:

1. The complete 4,208-point frame-zero source geometry defines a convex hull.
2. A deterministic 10 mm grid fills that hull with 5,620 internal particles.
3. The 4,208 benchmark identities are read out from eight-neighbour,
   inverse-distance material displacements; they are not MPM particles.
4. The frozen 16-query contact patch transfers to 25 finite-mass particles.
5. Contact applies a frame-rate-invariant compliant projection with fixed
   per-frame coupling `0.35`, rather than zeroing attached particle masses.

The particleization would have raised effective mass by about 6.2 times if it
retained density `1000 kg/m^3`. V2 therefore preserves the direct adapter's
`0.908928 kg` total mass. For 5,620 particles at 10 mm spacing this fixes the
density to `161.7309608540925 kg/m^3`. This keeps the experiment focused on
geometry and contact rather than quietly changing inertia.

## Public source and split

The source predictions and object observations were already opened by earlier
PhysTwin/MatPhys development work. Deterministic six-array comparator archives
preserve the incumbent and MatPhys float32 trajectories exactly. Their shared
action-support field is the incumbent's maximum predicted displacement,
normalized over material identities. Because matched comparator zero-action
rollouts are unavailable, their zero-action fields are explicitly exact
persistence placeholders and are not used to score the comparators.

Frames 0--137 are the previously used prefix/training region and are not used
to select v2 parameters. The frozen v2 split is:

- fit: `[138, 167)`;
- validation gate: `[167, 177)`; and
- source future: `[177, 198)`, unopened unless validation passes.

The parameter bank crosses Young's modulus values `5`, `25`, `100`, and
`500 kPa` with damping `0.002` and `0.02`. The lower 5 kPa arm is declared
because v1 selected the softest available value; the 2 MPa arm is removed
because every stiffer v1 arm worsened. No v2 object-motion score informed this
bank.

## Gate

Fit selection minimizes the equal-weight mean of identity-RMSE and Chamfer
ratios against exact persistence. The selected arm must then:

1. improve the balanced validation score over persistence by at least 5%;
2. remain within 10% of the incumbent on identity RMSE and Chamfer separately;
3. keep maximum zero-action drift below 2 mm;
4. replay to coordinate RMSE at most `1e-7 m`;
5. retain all eight candidates in the successful denominator; and
6. have final ensemble spread between `0.1` and `100 mm`.

Failure copies the incumbent physical archive byte for byte and forbids source
future scoring. MatPhys remains a comparator, not a selection candidate.

## Target-free feasibility

At clean implementation commit
`f2c5c9a73516d6bfa760c6e64f0a0dbaa68813f6`, a three-frame GPU run using
frame-zero geometry and the known controller action produced exactly 5,620
internal particles and 25 transferred contact particles. Driven, zero-action,
and replay runs were finite; zero-action drift and replay RMSE were zero, and
the maximum action response was `1.3952209847 mm`. No source metric outcome,
future source split, target, or held-out artifact was read.

The exact protocol, input hashes, simulation settings, parameter bank, and
gate are frozen in
`configs/sota/newton_mpm_double_stretch_zebra_volumetric_source_v2.json`.
