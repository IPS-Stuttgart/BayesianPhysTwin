# Tracking Cloth cost-aware support-robust action audit V2

## Why V2 exists

The first source feasibility audit used the registered probe cost inside the
exact contingent-plan certificate, so probe selection itself was cost-aware.
Its empirical source-gain summary, however, compared only the terminal physical
action loss against fallback. That summary could overstate the value of sensing.
V1 is retained unchanged as a transparent development result.

V2 scores a sensing decision by the same complete objective used conceptually by
the policy:

```text
complete sensing loss
  = terminal physical-action loss
  + probe cost * source task-loss scale.
```

The scale is the registered 90th percentile of all source physical-action
losses. Both terminal-only and complete-objective gains are reported so the size
of the correction is visible.

## Real-data finite decision

The official Tracking Cloth self-collision factorial contains, for every
material and repetition, three physically executed configurations:

1. `four_corners_normal`;
2. `four_corners_parallel`;
3. `two_corners_normal`.

They form the finite terminal action roster. The task loss is normalized
self-contact penetration RMS plus one quarter of the 95th-percentile cloth-edge
strain. Repetitions 1 and 2 are the only numerical source outcomes used here.
Repetition 3 remains reserved and closed.

A probe is represented by the binary source-frozen threshold of the causal
prefix shape-change feature from one of those configurations. The probe and
terminal action correspond to separate recordings in the same
material/repetition block. Consequently, this is a **reset-block feasibility
model**, not evidence that a probe can be inserted into and followed by an
action in one uninterrupted cloth episode.

## Incomplete and misspecified support

For each complete direct or contingent plan, V2 applies the exact
at-most-`epsilon` support-miss envelope

```text
Delta_epsilon(p,b)
  = Delta_0(p,b)
  + epsilon * max(0, upper[p] - lower[b] - Delta_0(p,b)).
```

The registered sensitivity grid is `epsilon in {0, 0.05, 0.10, 0.20}`. The
primary source gate uses `epsilon = 0.10`. Unknown terminal losses are assumed to
lie in `[0, 2]` after source normalization. These numbers are declared
ambiguity-set parameters; the experiment does not estimate or calibrate them.
Every selected plan records its maximum admissible support-miss probability.

## Source gate

The primary `epsilon = 0.10` setting must satisfy all of the following:

- at least two distinct source-optimal physical actions exist;
- at least two materials have the same optimum in repetitions 1 and 2;
- the selected primary policy uses a decision probe for at least one material;
- complete-objective source loss, including probe cost, improves on the exact
  fixed fallback.

A positive gate permits only preparation of a separately reviewed repetition-3
protocol. It does not launch target prediction or scoring.

## Reproduction

```bash
PYTHONPATH=src:. python -m \
  experiments.tracking_cloth_action_feasibility_costed_v2.run \
  --dataset-root \
    /home/github-runner/.cache/datasets/tracking-cloth-deformation-v1-zenodo-14644526 \
  --output /tmp/tracking-cloth-action-costed-v2
```

## Claim boundary

This audit establishes at most source feasibility of a finite reset-block
act--sense--fallback formulation under a declared loss box and support-miss
sensitivity. It does not validate the probe as nondestructive, establish reset
or counterfactual exchangeability, estimate `epsilon`, validate the unknown-loss
box, expose repetition-3 performance, demonstrate online robot execution,
calibrate uncertainty, authorize deployment, or certify safety.
