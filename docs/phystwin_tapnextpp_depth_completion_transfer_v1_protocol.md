# TAPNext++ RGB-D completion source-transfer protocol

## Purpose

The exploratory `single_lift_cloth` smoke showed that a source-qualified
single-camera RGB-D lift can complete a strict TAPNext++ multiview carrier
without weakening its accepted rows. It raised scored prefix support from
68.42% to 100% and reduced identity RMSE from 35.56 mm for persistence to
4.70 mm. That result is one opened source interaction, not transfer evidence.

This protocol freezes an eight-case opened-source panel before running any new
TAPNext++ predictions. It asks only whether the observation provider transfers.
It does not evaluate a Bayesian-PhysTwin future rollout.

## Frozen construction

For every fixed case, a 20-frame window is selected wholly inside the released
training prefix. Selection uses only the selected physical forward trajectory:
64 deterministic frame-zero farthest-point nodes are scored by endpoint RMS
displacement, with the earliest numerical tie retained.

At the selected source frame, up to four finite benchmark material identities
are chosen by deterministic farthest-point coverage. Their source-frame
positions initialize TAPNext++; all remaining manual prefix coordinates are
staged separately and cannot be opened until the strict prediction and the
depth-completed carrier are sealed.

The strict three-view carrier is unchanged wherever it has support. Accepted
strict rows qualify at most one camera using centered carrier agreement and an
overlap penalty. That camera may fill abstentions through sensor-depth lifting.
No camera precisions are multiplied. Completion covariance includes local RGB-D
uncertainty, carrier disagreement, and a 5 mm shared-bias floor. Rejection is
exact fallback to the strict carrier.

## Gates

A case passes only with at least 85% support, at least 10% gain over exact
persistence, identity RMSE at most 15 mm, and endpoint RMSE at most 15 mm.

The panel advances only when all aggregate gates pass:

- at least 6 of 8 fixed cases pass;
- aggregate supported fraction is at least 85%;
- case-balanced relative gain is at least 10%;
- case-balanced identity RMSE is at most 15 mm.

Technical failures and missing evaluations remain failed fixed cases. They are
never replaced. Passing authorizes only a separately frozen assimilation study
on already-open source cases.

## Claim and causal boundary

Manual material identity is used only for benchmark query initialization and
sealed prefix competence scoring. No future manual trajectory, future RGB-D,
future simulator metric, held-v8 artifact, or independent target is available
to provider prediction or selection. The physical-window selector reads a
forward-model trajectory, not a real future outcome.

Consequently, a positive result would establish transferable source-prefix
observation competence. It would not establish Bayesian-PhysTwin improvement,
independent confirmation, or state-of-the-art performance.
