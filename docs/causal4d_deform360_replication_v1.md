# Deform360 Shared-Physics Replication V1

Status: preregistered before access to media for the six selected objects.

This protocol is the independent public-data replication of the
deform360-001-rope-public-v1 pilot. It tests whether source-pooled physical
parameters transfer better than parameters selected from one source episode,
whether a causal contact-transition model improves over a static prefix contact
state, and whether uni- and bimanual interactions exhibit different behavior.

The executable lock is
configs/causal4d_public/deform360_replication_v1.json. Its canonical checksum is
validated by causal4d_public.deform360_replication before any replication
command can run.

## Claim Boundary

The pilot used a reduced, gridded, inextensible centerline forward model. It did
not run the official PhysTwin/Warp simulator and did not establish a
Bayesian-PhysTwin result. Its sealed target result only shows that a
source-pooled physical forward model beat constant persistence on one rope
episode.

The replication therefore separates two claims:

1. Shared-parameter transfer: pooled source fitting must beat the median of
   otherwise identical single-source fits on independently locked targets.
2. Backend competence: official PhysTwin/Warp is admitted only if it passes the
   source-only feasibility gate. Passing that gate establishes feasibility on a
   21-node public rope graph, not dense PhysTwin reconstruction.

Deform360 provides a vision-recovered robot trajectory rather than separate
commanded and measured actuator streams. Controller gain and delay are therefore
not identifiable here. Tactile is treated primarily as contact-timing
supervision and as a post-seal oracle, not as a modality presumed to improve
geometry prediction.

## Metadata-Only Selection

The dataset is pinned to revision
7fea8e20231a47641d1d2bc8791920ec4e62ec5e. Candidate objects were assigned to
filament, sheet, and volumetric strata from repository metadata. Within each
stratum, objects were ranked by SHA-256 using the fixed seed
deform360-shared-physics-replication-v1; the first two were selected.

The six locked objects are:

| Stratum | Object | Source episodes | Calibration episodes | Target |
| --- | --- | --- | --- | --- |
| Filament | 002-rope-silk | 0, 2, 5, 6, 7, 9 | 3, 4, 8 | 1, lift middle |
| Filament | 081-stripe-rope | 1, 3, 4, 6, 7, 9 | 0, 2, 8 | 5, lift both edges |
| Sheet | 085-scarf-cloth | 1, 3, 4, 6, 8, 9 | 0, 5, 7 | 2, lift center |
| Sheet | 083-blanket-cloth | 1, 2, 4, 5, 8, 9 | 0, 3, 6 | 7, lift adjacent edges |
| Volumetric | 092-squirrel | 0, 4, 5, 7, 8, 9 | 2, 3, 6 | 1, lift tail |
| Volumetric | 170-spider | 0, 1, 3, 5, 8, 9 | 2, 4, 7 | 6, bend in |

Targets are balanced at three unimanual and three bimanual interactions. The
split is a deterministic metadata-only hash decision. No selected-object image,
video, geometry, tactile outcome, or target metric was inspected before this
lock.

## Method Arms

Every admitted backend is evaluated with these seven arms:

1. Constant persistence.
2. One source episode selected on the same 200-candidate grid.
3. All six source episodes pooled on that grid.
4. Vision and released robot trajectory with a static contact state.
5. Prefix tactile with that contact state held static through the future.
6. A causal contact-transition policy.
7. Full future tactile oracle, evaluated only after target predictions are
   sealed.

The single-source result is the median target metric over every quality-passing
single-source fit. Target errors never choose the source episode.

The transition policy is a regularized discrete-time logistic hazard over each
gripper's active-contact state. Its causal features are current contact state,
current gripper openness, gripper-to-predicted-object proximity, and relative
closing speed. Coefficients are fitted on source episodes and hyperparameters
are selected on calibration episodes. Future target tactile is forbidden.

## Primary Gates

The pooling claim requires pooled fitting to beat the single-source control on
at least four of six targets, improve median target Chamfer by at least five
percent, and avoid median degradation in every object stratum.

The transition claim requires wins over static tactile contact on at least four
of six targets, no more than two percent median Chamfer degradation versus the
visual-static arm, and better contact-onset Brier score than static tactile
contact.

The replication unit is one object-target action. All six paired effects, their
exact signs, and their median are reported. Point coordinates and video frames
are not treated as independent samples.

## Official Warp Admission Gate

The official PhysTwin repository is pinned to commit
2b6630528141b9cba5a7677c8b88b2129b4a8390. Admission is tested only on
001-rope source episodes 0, 3, 4, 5, and 8. Episodes 1, 2, 6, 7, and 9 are
forbidden; in particular, the exhausted pilot target episode 6 cannot influence
the gate.

The gate requires deterministic repeated rollouts, finite states, plausible
edge strain, at least five percent source Chamfer improvement over persistence,
and leave-one-source-out wins on at least sixty percent of held-out source
episodes. A failure excludes official Warp from target replication without
changing the reduced-model replication.

## Information Order

1. Validate and commit this metadata-only lock.
2. Run the official-Warp gate on the allowed 001-rope source artifacts.
3. Access source observations for the selected objects and run source QA.
4. Use calibration episodes only for declared hyperparameters.
5. Construct each target prefix and seal all non-oracle predictions.
6. Verify prediction hashes.
7. Access target future geometry and compute metrics.
8. Access future tactile and compute the oracle comparison.

Any failed source QA may exclude an object only under a rule declared in the
lock. Outcomes cannot be used to replace a selected object or target.

## Source Geometry QA

The post-preregistration source-only camera policy is frozen separately in
`configs/causal4d_public/deform360_replication_source_qa_v1.json`. It reads the
first frame of one declared source episode per object, uses a pinned SAM 2.1
automatic mask from `brics-odroid-001_cam0` as an appearance anchor, and rejects
other views that do not produce a reference-consistent mask. A calibrated 3D
leave-one-view hull gate then selects reliable cameras. The final 12 cameras per
object are chosen by deterministic farthest-point sampling over camera centers,
with the reference camera retained.

This QA read no calibration-split or target episode and computed no prediction
metric.
All six objects passed with 23--27 cross-view-consistent cameras. The locked
artifact is archived under
`milestones/deform360-replication-source-qa-v1/`.
