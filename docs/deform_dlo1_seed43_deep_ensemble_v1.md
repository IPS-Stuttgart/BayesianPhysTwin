# DEFORM DLO two-seed candidate v1

This source-only route tests whether independent optimization uncertainty adds
useful predictive diversity beyond the checkpoint posterior. It trains a
second DLO1 model with seed 43 while preserving the seed-42 split, horizon,
batch size, optimizer, action information, 6,400-update budget, and checkpoint
schedule.

After both frozen runs finish, each seed contributes only its validation-
selected checkpoint. Two preregistered predictive means are compared:

1. equal member weights;
2. validation-error softmax weights with a fixed 1 mm temperature.

The comparison baseline is whichever individual seed has lower validation L1.
The ensemble must improve validation L1 by at least 1%. The selected candidate
is then evaluated on the already-open eight-case DLO1 source split and must
improve by at least 1% with at least five wins. Failure produces the exact
better-member fallback and prohibits fresh DLO2 ensemble work.

If the gate passes, DLO2 repeats seeds 42 and 43 from scratch with the same
registered operators. No DLO1 outcome may change the arm bank, thresholds, or
fresh protocol. Official DLO evaluation remains closed throughout this source
study.

The route is complementary to checkpoint averaging: checkpoint members share
one optimization trajectory, while the second seed probes between-run model
uncertainty. Its extra compute is justified only if that independent diversity
survives both validation and source-transfer gates.
