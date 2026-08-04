# Deform360 metadata-only preflight

## Purpose

The mounted path `/home/github-runner/.cache/datasets/deform360` contains the
public Deform360 release together with derived annotations and outputs from
earlier source experiments. Before any new evaluation opens an NPZ, NPY, HDF5,
PLY, image, video, tactile stream, or target metric, this preflight constructs a
content-addressed inventory from path names only.

The workflow is
`.github/workflows/deform360-metadata-preflight.yml`. Its locked protocol is
`protocols/deform360_metadata_preflight_v1.json`.

## Why arbitrary discovery is unsafe

A recursive search for any array shaped like `(frames, points, 3)` can silently:

- mistake a prior experiment output for an official released annotation;
- count several archives from one object as independent objects;
- identify episode or diagnostic filenames as object identities;
- open a previously reserved target before a cohort is locked;
- depend on filesystem traversal order when a resource limit is reached;
- mix fixed-identity tracks, unordered point clouds, frame-zero states, and
  simulator trajectories under one metric.

The preflight therefore does not inspect array keys or values. It first resolves
object identities against the vocabulary already frozen by the v1 and v2
Deform360 protocols.

## Classification

Every recognized object is assigned exactly one class, in this precedence:

1. `reserved_target`;
2. `prior_calibration`;
3. `prior_open_or_reserved`;
4. `candidate_name_only`.

The inventory records episode identifiers visible in names, top-level cache
roots, numeric-file suffix counts, complete numeric path names, and filename-only
contract hints. A string such as `002-rope-silk-ep0004` is mapped to
`002-rope-silk`; a filename such as `002-016-candidate-sheet.jpg` is never
invented as an object.

## Information boundary

The preflight may read:

- repository protocol JSON;
- directory names;
- file names and suffixes;
- the directory hierarchy.

It must not read or hash dataset file contents. In particular, it does not load
NumPy archives, HDF5 datasets, point clouds, RGB, depth, tactile data, robot
state, future geometry, or score-bearing outcomes.

The result records both

```json
{
  "dataset_payload_opened": false,
  "reserved_target_outcomes_opened": false
}
```

and a canonical `inventory_sha256`.

## Gate for the actual evaluation

No public-data scoring workflow should run until a follow-up commit binds:

- the exact inventory identity;
- an explicit allowed object and episode cohort;
- an exact allowed path list;
- one adapter and representation contract per path family;
- exclusion of every `reserved_target`;
- object-balanced aggregation;
- time-gap semantics for sampled frames;
- validity-aware persistence and last-residual baselines;
- the frozen Bayesian method and all comparison settings.

This metadata preflight is therefore useful progress but is not itself a model
evaluation or an accuracy result.
