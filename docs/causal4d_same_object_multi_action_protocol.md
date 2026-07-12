# Causal4D Same-Object Multi-Action Real Protocol

Status: preregistered software protocol; physical acquisition has not started.

The authoritative design is
`configs/causal4d/sloth_multi_action_v1.json`, with canonical design SHA-256
`6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968`.
The generator and validator live in `causal4d.real_protocol`.

## Objective

This protocol is the first real-data test that can distinguish three questions:

1. Does the existing twin predict an untouched continuation of an observed
   action?
2. Do persistent actuation variables `phi` transfer to another action under
   the same physical grasp?
3. Does a new-contact intervention require a fresh event variable `kappa_cf`,
   rather than reusing factual contact?

The experiment uses the same physical sloth instance as `single_lift_sloth`.
That choice minimizes reconstruction risk and makes action/contact variation,
not a new object pipeline, the principal experimental change.

## Design

The protocol contains 18 grasp sessions and 36 command executions:

```text
3 contact regions x 4 command profiles x 3 replicate blocks = 36 executions
```

Each grasp session contains two command profiles. The gripper returns to the
neutral pose between them but does not release the object. Across the six
sessions at each contact, the design uses every unordered pair of the four
profiles exactly once. This complete-pair construction has three benefits:

- every contact/profile cell has three repetitions;
- every profile occurs once in each replicate block;
- each session gives one chronological same-grasp transfer, action A to B.

The six injected realization conditions are assigned to sessions and rotated
across contacts as a balanced incomplete block. Every condition covers all
four command profiles, with each profile appearing once or twice. Consequently,
`phi` is fixed within a same-grasp pair without omitting a motion direction for
any injected condition.

### Contact regions

| ID | Physical region |
| --- | --- |
| `left_forepaw` | Anatomical left forepaw/wrist |
| `right_forepaw` | Anatomical right forepaw/wrist |
| `upper_torso` | Between the shoulders, below the neck seam |

Before confirmatory collection, each region must be registered as a fixed
canonical PhysTwin node set. The node file, node count, twin identifier, and
their SHA-256 values belong in `object_registration.json`.

### Neutral state

The sloth is held in a standardized suspended neutral pose with at least
100 mm table clearance. The controller waits two seconds before each command.
Every command follows a minimum-jerk outbound motion, 250 ms hold, minimum-jerk
return, and 1.5 second post-return settling period.

The neutral state is accepted only when:

- end-effector reset error is at most 2 mm;
- initial object-state Chamfer distance is at most 3 mm;
- contact-centroid error is at most 5 mm;
- RGB-D/actuator synchronization error is at most 5 ms;
- no RGB-D frame is dropped.

A failed gate is recorded and excluded before target evaluation. It is not
silently repaired or deleted.

### Command profiles

| ID | Controller direction | Amplitude |
| --- | ---: | ---: |
| `lift_low` | `[0, 0, +1]` | 40 mm |
| `lift_high` | `[0, 0, +1]` | 80 mm |
| `lower_high` | `[0, 0, -1]` | 80 mm |
| `lateral_low` | `[+1, 0, 0]` | 40 mm |

This gives an amplitude contrast, a matched forward/reverse pair, and a
lateral action without changing the waveform duration.

### Realization conditions

| ID | Injected change |
| --- | --- |
| `nominal` | Gain 1.0, zero delay, unbiased frame, locked grip |
| `gain_low` | Gain 0.85 |
| `gain_high` | Gain 1.15 |
| `delay_2_frames` | 2 RGB-D frames, 66.7 ms at 30 Hz |
| `frame_pitch_pos_3deg` | +3 degree controller-frame pitch |
| `slip_low_force` | Gripper-force scale 0.55, bounded slip target |

Commanded controls and measured end-effector motion are separate required
streams. The known injected value does not substitute for measured actuator
motion. Without the latter, gain, delay, frame bias, and contact transmission
would remain confounded.

## Slip Go/No-Go Gate

The low-force slip condition is conditional on a separate pilot performed
before any confirmatory execution. It passes only if:

- at least five pilot executions are recorded;
- at least two registered contact regions are represented;
- at least four produce 5-15 mm material-relative slip;
- the slip displacement coefficient of variation is at most 0.35;
- no execution completely releases the object;
- force/torque or gripper normal-force data are retained.

If this gate fails, collection stops. A new protocol version without slip must
be issued before outcomes from the 36-run design are observed. Slip is never
replaced post hoc with a condition that happened to work.

## Acquisition Order

The JSON file locks a deterministic, contact-interleaved session order. It
cycles contacts while using a hashed ordering within each contact. This avoids
long same-contact runs and spreads wear, temperature, and calibration drift
across conditions. The two command orders inside sessions are counterbalanced.
The same order is flattened for operators in
`configs/causal4d/sloth_multi_action_v1_schedule.csv`.

For every execution, record:

- synchronized RGB-D and its frame/checksum manifest;
- commanded control trajectory `u`;
- measured gripper/end-effector trajectory on the same monotonic clock;
- measured gripper state;
- calibrated camera and controller frames;
- initial object-state estimate;
- exact gain, delay, and frame-bias injection;
- fixed contact-region annotation;
- reset metadata;
- wear-cycle count and elapsed experiment time;
- object and room temperature when sensors are available;
- force/torque or normal-force data for every slip execution.

All artifact descriptors contain a relative path, SHA-256, and byte count.
Timestamped streams additionally name their shared clock.

## Data Layout

Create the complete non-overwriting acquisition skeleton with:

```bash
causal4d-real-protocol scaffold \
  configs/causal4d/sloth_multi_action_v1.json \
  /path/to/causal4d-sloth-multi-action-v1
```

The result is:

```text
protocol.json
acquisition_schedule.csv
object_registration.template.json
slip_pilot.template.json
sessions/<session_id>/session.template.json
executions/<execution_id>/manifest.template.json
```

Templates are deliberately named `*.template.json`; they cannot pass as
completed data. Acquisition promotes them to `object_registration.json`,
`slip_pilot.json`, and `manifest.json` only after all required values and
checksums are populated.

Validate a complete acquisition, including every file hash, with:

```bash
causal4d-real-protocol validate-dataset \
  configs/causal4d/sloth_multi_action_v1.json \
  /path/to/causal4d-sloth-multi-action-v1
```

## Locked Evaluations

### A. Factual continuation

All 36 executions are included. Causal4D receives only the first six `O+`
frames for intervention abduction. Every later frame remains untouched until
evaluation.

Report nominal PhysTwin, Bayesian-PhysTwin with nominal `z`, and Causal4D
posterior metrics. Aggregate at execution level and retain frame-resolved
curves; point-frames are not treated as independent replicates.

### B. Same-grasp intervention prediction

Each of the 18 sessions contributes one chronological pair. Action A is the
source, action B is the target. Persistent `phi` transfers, and factual
`kappa` is reused because the grasp is never released. The target command can
change direction, amplitude, or both.

The reverse B-to-A comparison is not used because it would reverse the actual
information order.

### C. New-contact intervention prediction

The protocol contains 12 matched-command transfers. Each holds the realization
condition and command profile fixed while moving to another registered contact
region. `phi` transfers and `kappa_cf` is resampled. The source always precedes
the target in the locked acquisition order, and target observations are
retained only for evaluation.

Each of the three unordered contact pairs appears four times; each command
profile appears three times; and each realization condition appears twice.

### D. Cross-action/contact calibration

Twelve folds are fixed in advance, one for every contact/profile cell. In each
fold:

- the target is the three repetitions of one contact/profile cell;
- no source session contains the held-out contact;
- no source session contains the held-out profile, even as its paired command;
- eight source executions fit the model and discrepancy;
- four disjoint source executions calibrate coverage and any semantic beta;
- the three target executions remain untouched.

Across the 12 folds, each execution is an out-of-fold target exactly once.
Likelihood temperature, discrepancy scales, coverage transforms, and semantic
beta are refit or selected only from that fold's fit/calibration sets. Target
results never choose a shared hyperparameter.

The four calibration executions in each v1 fold support an exploratory frozen
transform and an out-of-fold target test. They do not pass the later
undercoverage audit's ten-independent-execution gate for a real calibration
claim. Before physical acquisition, a calibration-headline protocol must either
pre-register a pooled source transform that still excludes every target action
and contact, or issue a higher-repetition v2. The 36-run v1 remains valid for
factual, same-grasp, new-contact, and calibration-transfer diagnosis.

## Counterfactual Language

Real repetitions are not individual-level counterfactual ground truth. A
physical sloth cannot undergo actions A and B simultaneously under identical
unobserved disturbances. The real result is therefore described as:

> Held-out interventional prediction from matched initial conditions.

Exact individual counterfactual validation remains confined to the controlled
simulator, where exogenous conditions can actually be shared.

## Claim Gate

Thirty-six executions are a compact first protocol, not automatic evidence of
publication-scale generality. The real claim advances only if:

- all acquisition and slip gates were locked before target evaluation;
- factual improvement repeats across actions rather than one trajectory;
- same-grasp transfer exceeds nominal-`z` baselines;
- new-contact performance benefits from resampling `kappa_cf`;
- held-out coverage is reported across independent executions and is close to
  nominal without target tuning;
- replay/reset variance is included in uncertainty and effect intervals.

If the expanded intervention oracle remains weak while discrepancy dominates,
the protocol supports the model-discrepancy diagnosis rather than a claim that
intervention abduction is the main real-world bottleneck.
