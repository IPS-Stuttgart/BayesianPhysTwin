# Deform360 evaluator contract v1

## Purpose

Deform360 Table 4 reports ParticleFormer multi-episode future errors of
`0.051 m` Chamfer distance and `0.079 m` track error. The paper describes the
task, but the public repository currently releases the raw data and annotation
pipeline rather than the world-model split, evaluator, baseline code, or
checkpoints. In particular, the paper calls track error a mean squared error
without exposing enough executable detail to decide whether the table contains
a mean distance, root mean squared distance, or squared distance.

An independent score is still useful for method development. It is not a
protocol-matched state-of-the-art comparison. The typed contract in
`causal4d_public.deform360_sota_evaluation` makes that distinction executable.

## Refusal boundary

The checked-in template is deliberately
`unresolved-non-authorizing`. It records every missing field rather than
guessing it. A direct Table 4 authorization requires all of the following:

1. the complete ordered fit and held object/episode split;
2. the exact evaluation horizon and stride;
3. checksummed material-particle identities for every held episode;
4. explicit, separate Chamfer and track visibility policies plus metric and
   aggregation definitions;
5. an author-released evaluator revision and entrypoint checksum;
6. exact reproduction of the published ParticleFormer `0.051/0.079` row.

`authorize_deform360_table4_claim` raises before comparing numbers unless the
contract has `official-parity` status and all six conditions pass. Merely
placing our score below the published numbers under an independent protocol is
not sufficient.

The scorer also records the evaluated frame indices and material-identity hash.
Chamfer and identity-based track error have separate visibility policies; a
track-confidence mask cannot silently remove geometry from Chamfer evaluation.
This prevents a changed horizon, unordered Chamfer-only particle cloud, or a
different visibility subset from silently replacing the declared experiment.
Panel aggregation is explicit and supports object-balanced evaluation so that
objects with more episodes do not become accidental extra replicates.

## Development mask and processing path

The locked Hugging Face revision was inventoried recursively before scaling the
experiment. It contains 179,700 tree entries: 179,698 below `raw/`, plus the
repository README and `.gitattributes`. No `processed/`, `pcd_clean`, tracking,
particle-identity, or evaluator artifact is present. The official release README
also states that processed annotations are not assumed to be present. Therefore
there is no public precomputed supervision path that can silently stand in for
the released annotation pipeline.

The official annotation pipeline uses gated SAM3 masks. On 2026-07-17 the
server had no authenticated SAM3 checkpoint, so a pinned public SAM2.1 fallback
was exercised as a development producer only:

- object: `004-rubber-band`;
- fit episode: `1`;
- camera: `brics-odroid-001_cam0`;
- SAM2 repository revision:
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`;
- checkpoint SHA-256:
  `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`;
- propagated frames: `317/317`;
- empty masks: `0`;
- mask area: `11,113` to `18,711` pixels, median `15,569`;
- output HDF5 SHA-256:
  `360ceff9292d41f0184bd149a72748739d7edbe473aa27d3c0560f20e4dc7662`.

The frame-zero mask and five evenly spaced propagation frames were visually
checked. The rubber-band bundle remains selected through gripper occlusion and
returns to the unoccluded object at the final frame. A source-only multiview
geometry audit then accepted 21 of 32 calibrated cameras. Full propagation over
those views produced 6,657 camera-frames; 19 cameras had no empty masks and two
retained explicit empty-mask intervals as weak/occluded views. The checksummed
mask-panel result is
`278d751687ecf52b60509abc09c060e86b4c3aaba6b8d04a37fb6ee4380bd451`.

The accepted views were staged without modifying the aligned release data and
passed a frame-zero smoke through the released Deform360 processing code at
revision `0fe36f0b7a7a917ba62b5f8cee707299a9a4a317`. The official Splatfacto path
exported 12,296 Gaussians, with PLY SHA-256
`583b988427221d1cfca785125666ae02ffa429ec48da15a661d80b7b4c8743e1`.
The official URDF and metric-depth stages then produced valid `(1,720,1280)`
artifacts for all 21 cameras. The reconstruction used only 200 optimization
iterations and is therefore a dependency and contract smoke, not a quality
result. This establishes development-pipeline feasibility, not equivalence to
Deform360's SAM3 masks, published particle annotations, or evaluator.

Subsequent development episodes use
`scripts/remote/stage_deform360_sota_development_episode.py`. The runner accepts
only objects in the locked 12-object development panel, enforces the fit versus
held-development episode split, verifies the episode-1 reference-mask hashes,
and re-identifies each same-view object mask from the new episode's frame zero.
Future frames are used only to propagate the development annotation; no outcome
metric or simulator residual enters mask selection. It then builds a symlink-only
view for the released Deform360 processing code and records checksummed mask and
staging artifacts. Confirmatory objects are rejected before any media access.

After reconstruction, the development-only observation runner
`scripts/remote/run_deform360_sota_development_observations.py` executes the
released depth, CoTracker, advected-point-cloud, and control-point stages at
their pinned revisions. Its manifest hashes every stage, defines material
identity as the ordered frame-zero seed points, verifies that the identity
count is unchanged through the rollout, and records the actual point horizon.
No prediction metric is computed by this step.

`build_development_evaluator_contract` accepts only checksummed
`held-development` manifests. The evaluation start, stop, and stride must be
declared explicitly and must fit every included episode. It fixes Euclidean
symmetric Chamfer, identity-aware mean Euclidean track error, all-finite point
visibility, and object-balanced aggregation. The resulting contract remains
`independent-protocol`: even a score below the published ParticleFormer row is
programmatically refused as a direct Table 4 claim until author-evaluator parity
is established.

## Next evidence

The next source-only steps are:

1. run the released reconstruction at declared development iterations, then
   depth, tracking, point-cloud, and control-point stages on fit episode 1;
2. instantiate an `independent-protocol` evaluator contract from checksummed
   held-development manifests and a prospectively declared horizon;
3. run persistence and reusable-PhysTwin smoke scores;
4. replace the independent contract with an official one only if the authors
   release the split/evaluator or provide enough artifacts to reproduce the
   ParticleFormer reference row exactly.

No confirmatory object, held future, PokeFlex target, or frozen Causal4D
artifact is opened or modified by this work.
