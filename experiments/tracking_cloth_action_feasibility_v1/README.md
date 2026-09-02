# Tracking Cloth physical-action source feasibility v1

This experiment asks whether the public Tracking Cloth self-collision factorial can
support a later **act--sense--fallback physical-twin** study with genuine physical
action alternatives.

The self-collision panel contains four materials, three physically executed
release configurations, and three repetitions. Repetitions 1 and 2 are opened as
source data. Repetition 3 remains numerically reserved.

## Physical decision

Each material/repetition block contains the complete action set:

1. `four_corners_normal`;
2. `four_corners_parallel`; and
3. `two_corners_normal`.

Because all three releases were physically executed in every source block, the
source action-loss vector uses measured outcomes rather than model-generated
counterfactuals.

The registered task loss is

```text
contact-depth RMS
  + 0.25 * 95th-percentile grid-edge strain
```

after the first 0.5 seconds. It is normalized by the initial cloth diameter.

## Decision-directed probe surrogate

The first 0.5 seconds of each interaction provides one source probe feature:
normalized pairwise cloth-shape change. Source medians convert each feature into
a deterministic binary outcome. The exact
`act_sense_fallback_certificate_v1` then evaluates:

- direct physical release choices;
- resettable probe-then-release contingent plans; and
- exact fallback to the registered conservative release.

This is only a **resettable block-level feasibility model**. The recordings do
not demonstrate that the cloth can be returned to an identical state after a
probe, nor that the probe and terminal action were executed online in one trial.

## Timestamp contract

The publisher's nominal capture rate is not treated as an exact timestamp
lattice. The source loader:

1. requires monotone recorded timestamps;
2. finds one complete initialization frame within 0.25 seconds;
3. causally carries isolated missing marker observations after initialization;
4. linearly resamples the measured source trajectory onto a registered 30 Hz
   analysis grid; and
5. records native minimum, median, and maximum time increments in the evidence.

This avoids repeating the earlier technical failure caused by requiring every
recorded increment to equal exactly `stride / 120`.

## Source-only promotion rule

A separately reviewed repetition-3 protocol is considered only if source data
show all of the following:

1. at least two distinct source-optimal physical actions;
2. a stable optimal action across repetitions for at least two materials; and
3. at least one exact certificate setting that uses a decision probe and
   improves source loss over the fixed fallback.

Failure of any item terminates or redesigns this benchmark without opening
repetition 3.

## Information boundary

The dataset byte inventory may hash all public files, but numeric trajectory
access is restricted in code to repetitions 1 and 2. Repetition 3 is rejected
before `_row_stream` is called. The workflow uploads only compact source metrics,
the protocol, and a summary; it does not upload trajectories.

A positive source result would justify only a new, separately frozen target
protocol. It would not establish held-out physical-action benefit, online active
sensing, reset fidelity, unseen-material transfer, calibrated decision risk,
closed-loop robot control, deployment safety, or state of the art.

## Run

The permanent workflow is:

```text
.github/workflows/tracking-cloth-action-feasibility-v1.yml
```

Pull requests run contract, lint, formatting, and target-closure checks on
GitHub-hosted runners. A merge to `main` runs the source-only audit on
`[self-hosted, Linux, X64, gpuserver4090]`.
