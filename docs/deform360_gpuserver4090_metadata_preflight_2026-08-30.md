# Deform360 gpuserver4090 metadata preflight

Date: **2026-08-30**

## Purpose

This record establishes the metadata-level state of the resumable Deform360
staging tree at

```text
/mnt/seagate10tb/florianpfaff/datasets/deform360
```

before designing the public-data validation for query-identifiable physical
belief revision. The preflight was triggered by committed GitHub Actions workflow
files and ran on the self-hosted runner carrying the `gpuserver4090` label.

It did not open NumPy arrays, HDF5 datasets, point clouds, images, videos,
tactile streams, robot states, future geometry, or score-bearing target
outcomes.

## Runner binding

GitHub assigns the custom label `gpuserver4090` to a runner registered as
`workstation1`. The workflow therefore binds the execution by the custom label,
the exact mounted root, and the observed GPU capability rather than by hostname
alone.

The successful run recorded:

```text
runner label:  gpuserver4090
runner name:   workstation1
hostname:      workstation1
GPU 0:         NVIDIA GeForce RTX 4090
GPU 1:         NVIDIA GeForce RTX 4090
data root:     /mnt/seagate10tb/florianpfaff/datasets/deform360
```

The initial hostname-only attempt failed closed before dataset traversal. It was
not interpreted as a data failure.

## Names-only inventory

Successful workflow run: `33300946082`

Artifact: `9728916412`

Artifact SHA-256:

```text
475a74ef888bff4bc1709cc04cd88a9e4bc282914a7c254f13f071ad09053f17
```

The mounted staging tree contained:

| Quantity | Value |
| --- | ---: |
| Named files | 514,121 |
| Named directories | 16,667 |
| Total named file bytes | 216,903,070,312 |
| Objects recognized by the existing v1/v2 protocol vocabulary | 168 |
| Recognized objects with numeric paths | 168 |

Two complete names/size/timestamp snapshots were taken approximately 26 seconds
apart. File count, directory count, total named bytes, and the aggregate
path-size-mtime identity were unchanged during that interval.

The existing protocol classified the 168 recognized objects as:

| Existing classification | Objects |
| --- | ---: |
| Candidate by name only | 104 |
| Prior calibration | 12 |
| Prior open or reserved | 40 |
| Reserved target | 12 |

The names-only content identity is:

```text
cb37c193052baae41140d0cb48c642c16e3840a850ca104a4883555eb19f7611
```

The execution-bound inventory identity is:

```text
8cbbae7c0dca51f4dc019d8ceb6fc5543e9ccb5f7e94bec0859e4bdd8005f887
```

The twelve previously registered target objects were present by name. Their
presence does not authorize opening them.

## Downloader status boundary

Successful operational-status run: `33301128632`

Artifact: `9728965070`

Artifact SHA-256:

```text
a759fd0c33a7b44306504157f5d578dc90aefc4e8241ecd744bf2654ac338a9d
```

`download.status` exists but is mode `0600`, owned by UID/GID 1002, and is not
readable by the GitHub runner account. The workflow did not use `sudo`, change
permissions, impersonate the owner, or otherwise escalate privileges.

Six files ending in `.incomplete` were present under the Hugging Face cache for
`001-rope/brics-odroid-028_cam0`. Every marker had size zero and was approximately
ten hours old at capture. No nonzero partial payload was found. The downloader
lock file was also approximately ten hours old. These observations are
consistent with stale cache markers, but the unreadable status file prevents a
formal assertion that the downloader itself declared completion.

## Decision

The retained decision is:

```text
stable-staging-tree-completion-not-certified
```

The tree is large, stable over the observed interval, and contains 168 objects
recognized by prior protocol vocabularies. It is suitable for the next
**metadata-only** preparation stage. It is not yet authorized for public-data
scoring because completion, payload validity, and a disjoint confirmatory cohort
have not been established under the new paper's information boundary.

## Next gate

Before any target-bearing payload is opened, a separate committed contract must:

1. construct the complete cross-project exclusion union from BayesianPhysTwin,
   Causal4D, Prob4D, and paper evidence;
2. freeze exact calibration and evaluation object/episode identities;
3. exclude all previously reserved targets and all objects previously used for
   method development or reported results;
4. bind an exact allowed path list and one representation adapter per path
   family;
5. freeze query definitions, quotient construction, tolerances, guards,
   comparison arms, metrics, and object-level statistical units; and
6. seal all source-derived predictions before opening target futures.

The 104 `candidate_name_only` objects are candidates for that exclusion-union
analysis; they must not yet be described as fresh or confirmatory.

## Claim boundary

This is metadata and workflow evidence only. It does not establish download
completion, official release completeness, payload integrity, model competence,
query-identification accuracy, calibrated uncertainty, real-data improvement,
physical transport, or permission to inspect reserved target outcomes.
