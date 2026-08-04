# Deform360 source-hull contract probe

## Purpose

This probe is the first payload-metadata step after the names-only Deform360
preflight. It is deliberately narrower than a model evaluation. The exact cohort
was committed before the probe opened any archive member.

The cohort contains 36 `sampled_hulls.npz` archives from six already-open source
objects:

- `002-rope-silk`;
- `081-stripe-rope`;
- `083-blanket-cloth`;
- `085-scarf-cloth`;
- `092-squirrel`;
- `170-spider`.

Every entry is classified `prior_open_or_reserved` by the frozen v1/v2 object
vocabulary. No reserved-target object is admitted. The exact object, episode,
and path list is bound in
`protocols/deform360_source_hull_contract_probe_v1.json`.

## What the probe reads

For each locked archive, the probe reads:

- ZIP member names;
- `frame_indices` integer values;
- `point_offsets` integer values;
- the NumPy header of `points_world_m`;
- complete compressed archive bytes for a cryptographic SHA-256 digest.

The complete-byte hash means the coordinate payload is cryptographically bound,
but the coordinate values are not decoded or inspected. No RGB, depth, tactile,
robot-state, prediction, future score, or target metric is read.

## Fail-closed contracts

Every archive must have:

- unique ZIP member names;
- `frame_indices.npy`, `point_offsets.npy`, and `points_world_m.npy`;
- at least three nonnegative, strictly increasing frame indices;
- offsets starting at zero and increasing strictly;
- one nonempty point set per sampled frame;
- a final offset equal to the number of point rows;
- floating point data with header shape `(N, 3)`.

A missing archive, path escape, duplicate member, malformed array, empty frame,
unsupported NumPy header, or identity mismatch stops the workflow.

## Why cadence is checked before scoring

The earlier generic evaluator treated every stored hull index as one equal time
step. That is invalid when `frame_indices` have irregular gaps. This probe reports
the complete stride distribution for every episode before a prediction method or
baseline is selected.

A later source-only diagnostic may proceed only after it commits one of these
policies:

- constant-stride propagation using the verified common stride; or
- explicit gap-aware propagation using the actual frame differences.

The choice must be committed before any coordinate values are decoded and before
any prediction error is computed.

## Evidence boundary

A successful probe establishes only that the exact source archives are structurally
usable and identifies their sampling cadence. It does not establish:

- prediction accuracy;
- physical-state correction;
- calibrated uncertainty;
- official Deform360 benchmark parity;
- transfer to a new object;
- any reserved-target result.

The next legitimate step is a separately locked, object-balanced source diagnostic
using the exact file hashes and cadence policy produced by this probe.
