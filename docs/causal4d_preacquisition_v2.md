# Causal4D Pre-Acquisition Amendment v2

## Why an amendment is required

The original 36-execution protocol remains the confirmatory contact/action
grid, but it cannot support every newly locked statistical claim by itself:

- no contact/profile/realization cell has three exact repeats;
- all command profiles use a 0.75 s outbound duration and a 0.25 s hold;
- each original calibration fold has only two independent calibration sessions;
- 90% execution-level split conformal needs at least nine calibration units to
  produce a finite order statistic without an infinite sentinel.

`configs/causal4d/sloth_multi_action_v1_power_audit.json` records these facts.
They were identified before physical acquisition and may not be repaired after
target outcomes are observed.

## Locked amendment

`configs/causal4d/sloth_preacquisition_v2.json` has canonical SHA-256
`57d9788c4de31ff3f103d487fcf7b2080523e69ead402e38961f47d0e749a719`.
It leaves all 36 confirmatory executions and target IDs unchanged.

Before those executions, collect a 12-execution source-only panel at the
registered upper-torso contact. Four profiles each receive three fresh-reset,
fresh-grasp repetitions:

| Profile | Purpose |
| --- | --- |
| `lift_high` | common reference and exact repeatability |
| `lower_high` | matched direction reversal |
| `lift_high_slow` | matched speed contrast at fixed amplitude |
| `lift_high_long_hold` | matched hold-duration contrast |

The panel estimates the empirical reset/actuation noise floor and tests the
direction, rate, and relaxation signatures. It is forbidden from entering a
confirmatory target fold. Multi-camera leave-one-view-out transfer supplies the
observation-bias contrast without adding another action cell.

## Analysis lock

The replication unit is the grasp session. Accuracy intervals use an
equal-session paired cluster bootstrap with 20,000 replicates and seed
`20260712`. Residual-on-covariate regressions use a CR1 session-clustered
sandwich covariance. Point-frames and coordinates are never treated as
independent replications.

Every physical candidate is evaluated both alone and with the same prefix-only
readout-persistence correction refitted on top. Promotion requires accuracy
and calibration gates plus shrinkage of the correction field: the upper 95%
session-cluster bootstrap bound on the mean log mechanism-to-nominal correction
RMS ratio must be below zero.

Each amended outer fold preserves the original three target executions. The
other 15 sessions are partitioned into six strict cross-contact/cross-profile
fit sessions and nine calibration sessions. Exactly one preregistered execution
per calibration session supplies the score

```text
q_0.90(abs(coordinate error) / raw predictive standard deviation).
```

The 90% split-conformal rank is therefore 9 of 9. This is finite but coarse and
must be reported as such. Target-specific scale adaptation remains forbidden.

## PyRecEst boundary

PyRecEst `2.4.1` is an optional Python 3.11+ dependency. The dry run uses
`pyrecest.calibration.fit_time_offset` and
`fit_sensor_bias_correction` for source-only command/measured-actuator
diagnostics. Hardware timestamps remain authoritative, and the affine bias fit
is not interpreted as a posterior over frame rotation, gain, slip, or material
lag.

The input NPZ contract contains:

```text
command_times_s
command_positions_m
measured_times_s
measured_positions_m
```

Run:

```bash
causal4d-calibrate-actuator-realization \
  /path/to/actuator_trace.npz \
  /path/to/actuator_realization.json \
  --execution-id dry-run-1
```

## Collection gate

The required sequence is:

```text
contact registration
-> slip pilot
-> 12-run signature/repeatability panel
-> actuator and support validation
-> one nonconfirmatory end-to-end dry run
-> freeze analysis implementation
-> unchanged 36-run confirmatory acquisition
```

Generate or validate the amendment with:

```bash
causal4d-preacquisition-protocol generate \
  configs/causal4d/sloth_multi_action_v1.json \
  configs/causal4d/sloth_preacquisition_v2.json

causal4d-preacquisition-protocol validate \
  configs/causal4d/sloth_multi_action_v1.json \
  configs/causal4d/sloth_preacquisition_v2.json
```

The first confirmatory execution remains blocked until every collection-gate
field is satisfied and recorded in a new immutable acquisition artifact.
