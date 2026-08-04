# Deform360 exploratory source-hull inventory v1

## Purpose

This inventory is the clean successor to the closed source-hull probe. It answers
one metadata-level question before any prediction method is defined:

> Which of the exact, already-open source archives contain enough nonempty sampled
> hulls for a later causal prediction protocol, and what cadence and missing-hull
> patterns would that protocol have to handle?

The study is explicitly **exploratory, non-fresh, and non-score-bearing**. The
mounted cache cannot provide untouched candidate-object trajectories: the
names-only preflight in PR #115 found official numerical data only for previously
used objects. This inventory therefore cannot establish independent validation.

## Bound source cohort

The protocol
`protocols/deform360_exploratory_source_hull_inventory_v1.json` binds:

- the final PR #115 metadata-preflight identities;
- six previously opened source objects;
- exactly 36 `sampled_hulls.npz` paths under the official release root;
- the upstream Deform360 revision; and
- a strict no-reserved-target information boundary.

There is no recursive discovery. A missing path, path escape, duplicate ZIP
member, malformed integer metadata, incompatible point-array header, or archive
hashing failure stops the run.

## Deliberately admitted source properties

Unlike the closed preregistered probe, this exploratory protocol declares these
conditions before execution and reports them rather than changing the contract
after observing them:

- a sequence may contain only one or two stored hull frames;
- `point_offsets` may repeat, which denotes an empty sampled hull;
- removing empty frames may change the effective frame-index cadence; and
- an archive may be structurally valid but prediction-ineligible.

An archive is only a **prediction candidate** when it contains at least three
nonempty sampled hulls. That label does not authorize coordinate decoding or
freeze a scoring cohort. Empty-frame handling, stride semantics, baselines,
methods, aggregation, and thresholds remain unresolved until a separate protocol
binds exact eligible archive hashes.

## Reads and prohibited operations

The evaluator may read:

- archive member names;
- integer `frame_indices` and `point_offsets` values;
- the NumPy header of `points_world_m`; and
- complete compressed archive bytes for SHA-256.

It does not decode any `points_world_m` coordinate, read RGB/depth/tactile data,
run BayesianPhysTwin, compute a prediction or score, or open a reserved target.
Every archive record repeats `coordinate_values_decoded=false`.

## Result interpretation

The output reports equal-path metadata, including:

- stored and nonempty frame counts;
- empty-hull frame indices;
- raw and nonempty-frame stride patterns;
- prediction-candidate paths and per-object counts;
- complete archive SHA-256 identities; and
- content-only and execution-provenance inventory identities.

A successful workflow means that the inventory completed under its declared
information boundary. It does not mean that all archives are prediction-eligible
or that a later non-fresh diagnostic is scientifically worthwhile. Those are
results of the inventory, not prerequisites silently imposed on it.
