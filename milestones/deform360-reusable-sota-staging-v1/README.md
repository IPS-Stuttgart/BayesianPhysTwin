# Deform360 reusable-PhysTwin SOTA staging v1

This milestone records the first input-side execution of the prospective
`deform360-reusable-sota-v1` protocol. It is an engineering and provenance
milestone, not a dynamics result or a state-of-the-art claim.

## Completed

- The public Deform360 snapshot is pinned to revision
  `7fea8e20231a47641d1d2bc8791920ec4e62ec5e`.
- Development object `004-rubber-band` was downloaded without audio. Its raw
  tree contains 832 files and has SHA-256
  `c2b8a096806d87cc5a0599fab88443c36343a923c30a1d97b9576f303ba6d937`.
- The official undistortion/alignment stage completed for all ten episodes.
  Every episode has 32 valid camera streams with a common frame count inside
  the episode.
- The official ArUco robot stage completed for the six registered fit episodes
  `1,3,4,6,7,9`. Safe `robot.npz` artifacts load with the expected frame count
  and unimanual/bimanual cardinality.
- No robot artifact was generated for held episodes `0,2,5,8`.
- No confirmatory object was downloaded or inspected.

The machine-readable checksums and frame counts are in
`artifacts/004-rubber-band-input-qa.json`.

## Interpretation

The public raw release is sufficient to reproduce calibration, alignment, and
robot-action recovery. It is not, by itself, sufficient for a direct
Deform360 Table 4 comparison. That comparison additionally needs the processed
dynamic geometry/particle annotations and the exact published object split,
forecast horizon, and evaluator. Those artifacts are not currently exposed by
the public baseline code used here.

The next valid computation is therefore a source-only annotation smoke test on
`004-rubber-band`, or ingestion of official processed annotations if the
authors provide them. A direct numerical SOTA claim remains forbidden until
evaluator parity is established.

## Claim boundary

This milestone establishes only that the locked public-data input path is
executable and reproducible. It does not establish reusable dynamics transfer,
calibration, superiority to ParticleFormer or PGRD, or any Causal4D result.
