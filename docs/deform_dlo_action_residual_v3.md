# DEFORM action-conditioned residual analog v3

## Purpose

The 6,400-update public DEFORM reproduction is already competitive on opened
DLO1 source data, but checkpoint averaging has less than one percent point
headroom. This source-only arm instead tests the discrepancy pattern supported
by the wider Bayesian-PhysTwin evidence: simulator residuals may transfer
between executions when conditioned on the realized action and current material
state.

For a query trajectory, the method constructs a rigid-invariant descriptor from
the observed initial DLO state and the complete known trajectory of the four
clamped action nodes. It retrieves whole-trajectory residual fields from the 40
fit trajectories using an RBF mixture. Exact duplicate descriptors are collapsed
before weighting, so repeated donors do not create artificial confidence. The
point correction is a validation-selected conservative shrinkage of the mixture
mean. Coordinate uncertainty is the unshrunk donor-mixture spread plus a fixed
metric floor; it is never divided by the number of dense coordinates.

The query's future unclamped nodes, state innovation, and outcome error do not
enter association or reliability. The four clamped nodes remain byte-identical
to the selected DEFORM baseline.

## Information boundary

DLO1 is post-open development data. The fit split estimates donor residuals, the
eight validation trajectories select one frozen arm, and the eight DLO1 source
trajectories are loaded only after the validation-selection seal exists. The
runner installs read guards for both the DLO1 official evaluation directory and
the entire DLO2 data tree. It contains no DLO2 execution path.

Passing the DLO1 source gate would authorize writing a separate, preregistered
DLO2 source protocol using the same descriptor, posterior, arm bank, and gates.
It would not authorize official evaluation directly. Failure keeps the exact
update-6400 DEFORM prediction and closes this method family on DLO1.

## Frozen gates

Validation requires at least one percent trajectory-balanced L1 improvement,
six of eight wins, and no case worse than 1.05 times baseline. The selected arm
then must independently achieve at least one percent improvement and six of
eight wins on the source panel, keep every case within 1.10 times baseline, and
reach at most the published 10.1 mm DLO1 reference.

This arm uses no Prob4D, camera observation, tactile stream, or target-frame
material observation. It is an identical-action-information discrepancy model,
not an online-assimilation benchmark.
