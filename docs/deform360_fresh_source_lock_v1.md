# Deform360 fresh source lock v1

## Purpose

This package prepares a genuinely fresh Deform360 object cohort without
reading future object positions or metrics. It binds the official annotation
release at commit
`d8522a4403b766aeb387510c04e89032a56fdf35` and the unchanged Bayesian update
at commit `e2f8d827bfd60df79eeffee511a5df7e2d53ea21`.

The package has three artifact types:

1. `Deform360FreshSourceAdmission` for one object episode;
2. `Deform360FreshObjectExclusionManifest` from an independent evaluation
   owner;
3. `Deform360FreshObjectCohortLock` for the final ordered physical-object
   cohort.

The preregistered source-lock configuration is
`configs/sota/deform360_fresh_object_source_lock_v1.json`, SHA-256
`fc8a68bd769a7b71d570e4c6879ea186b3ff4453457d29e4dc141a462a0038b7`.

## Admission boundary

Admission may inspect only:

- the canonical public raw object-directory and episode identity;
- the descriptive `metadata.object` label, without treating it as canonical;
- the exact `yes`/`no` bimanual metadata enum;
- source-stream provenance digests;
- control-point provenance hashes and counts;
- camera names;
- `split.json` indices;
- the vertex count in the frame-zero PLY header.

It hashes `final_data.pkl` to bind custody but never deserializes it. Invalid
pickle bytes are accepted by the test fixture when every source contract is
otherwise valid, proving that future geometry is not being loaded indirectly.

The default frozen gates are:

| Field | Requirement |
|---|---:|
| Calibrated camera count | at least 3 |
| Frame-zero points | 128 to 10,000 |
| Trajectory rows | exactly 76 |
| Online update frames | 19, 38, 57 |
| Future test rows | at least 8 |
| Split | contiguous released 80/20 rule |
| Contact window | exactly the 76 retained rows |
| Source identity | raw metadata parent directory equals the public object ID |

The last update must remain inside the train prefix.

An accepted artifact is checked again when it enters a cohort lock. Recomputing
the JSON digest after changing a gate is insufficient: accepted artifacts must
still match the frozen default configuration and satisfy every identity,
camera, point-count, row-count, contact-window, split, and stream-provenance
invariant.

## Released split hazard

The public control-point generator drops inactive contact frames when stacking
`final_data`, but writes `split.json.frame_len` from the undropped contact
window. Its own test does not compare those lengths. The admission artifact
therefore requires:

```text
split.frame_len == control_points.meta.outputs.num_active_frames
```

An episode that violates this relation is rejected before prediction. This is
a source-contract failure, not a model failure, and the case may not be
silently replaced after cohort lock.

## Exclusions

An evaluation owner supplies physical-object IDs to the exclusion builder.
The resulting artifact emits only namespaced SHA-256 values, not the IDs or
any target artifact. Cohort selection hashes each candidate object ID under
the same namespace and excludes matches.

This permits independent held-cohort ownership without granting this process
access to held outcomes.

The held-v8 exclusion is additionally backed by a committed-source history
audit. It covers all revisions that changed the production authorization and
source-preparation paths across the published v8 through v8.3 lineage. The
audit accessed neither the campaign root nor target, query, score, barrier, or
outcome artifacts. Its 23 case identities reduce to seven physical-object
hashes, all already subsumed by the Prob4D exclusion.

## Frozen staging queue

The first source queue is
`configs/sota/deform360_fresh_source_staging_queue_v1.json`, internal SHA-256
`f80fed80ca2b9f1857539834bd92c6acb1b45a88eefbcae16e35cddaf9185d0e`.
It binds the public catalog, metadata preflight, all known independent
exclusions, and 18 episode-zero candidates before candidate media or processed
geometry is inspected.

The queue contains five filament, seven sheet, and six volumetric morphology
strata. These labels are predeclared from public identity only and are used
solely for deterministic balance. If fewer than 12 candidates pass source
admission, the run stops and records every rejection. Any reserve candidates
require a new immutable queue locked before their payloads are inspected.

## Frozen source window

The queue-bound temporal rule is
`configs/sota/deform360_fresh_source_window_v1.json`, internal SHA-256
`015305926274bda59ae0b03390a86ac321e615b598001961fc70f13ee9f69511`.
It binds the completed camera-only source download before source RGB is decoded:

- download-manifest SHA-256
  `a7774030848e2df5d4f33de37d8b6292b79665914053d690eff37b0f56f958ff`;
- 18 objects, 1,452 files, and 1,834,930,956 bytes;
- an exact 12-camera panel;
- 81 staged frames and 76 prediction rows;
- candidate starts at frame 8 with stride 6;
- action scoring from staged step 19 through step 74;
- earliest-start tie breaking.

The score uses only the released end-effector translation
`robot.actions[...,0,:]` and robust gripper-closure confidence. The known future
action is an explicit conditioning input. Object geometry, tracks, response,
tactile, and target metrics are not used to choose the window.

This corrects a defect in the frozen negative selective-virtual-sensing
experiment: its legacy `select_action_only_window` averages the translation,
three rotation rows, and aperture metadata as if all five rows were spatial
points. That function remains unchanged for exact reproduction of the negative
result. New source windows use `select_fresh_source_window`.

Window selection is not an admission gate. In particular, no motion or response
threshold can remove a queued case or cause an implicit replacement. The
existing source contract alone determines admission, and a method may use a
source-locked exact fallback when its observation-support gate fails.

The automatic source-mask rule is separately frozen in
`configs/sota/deform360_fresh_source_masks_v1.json`, internal SHA-256
`1c530153e693149e3defc54d20c153dbaff2aa26009006f7c7a805ca7db0f67c`.
It binds the generic object-agnostic SAM2 selector, SAM2 commit and checkpoint,
the exact window implementation, 81 mask frames, and a minimum of eight
successful cameras. There is no manual prompting. Per-camera failures are
preserved, and fewer than eight successful masks is a technical failure rather
than permission to tune the selector or replace the object.

The completed mask campaign is sealed in
`results/sota/deform360_fresh_source_lock_v1/fresh_source_mask_campaign_v1.json`.
Fifteen of the 18 queued cases are ready for source processing. Three cases
(`047-rectangle-sponge`, `013-glove-cloth`, and `104-alloy`) produced only
seven successful camera masks and remain technical failures under the frozen
eight-camera minimum. They were neither repaired nor replaced.

The next source-only stage is frozen in
`configs/sota/deform360_fresh_source_processing_v1.json`, internal SHA-256
`3ba3931816d6cca5e9e25f82c7aee222972c4b2ad29cfd392bdd8225affcdad3`.
For each processing-ready case it uses all successful frozen-panel cameras in
lexical order and runs the pinned Deform360 reconstruction, gripper-mask,
depth, CoTracker3, point-cloud, and control-point stages in a derived
workspace. The sealed source windows and masks remain immutable. Processing
or admission failures are terminal for that queue entry; they do not authorize
camera changes or a replacement object.

The completed processing campaign is sealed in
`results/sota/deform360_fresh_source_lock_v1/fresh_source_processing_campaign_v1.json`,
internal SHA-256
`62a34f301439746172953a8216509925fd24bb6f0892afa1787c610be125a8c7`.
Fourteen cases were admitted. The three mask failures remain terminal, and
`006-fur` retains its first-attempt source-processing failure caused by a
missing CUDA-toolkit runtime binding. It was not retried or replaced. The
processed-source inventory contains 4,198 files and 1,230,127,897 bytes with
tree SHA-256
`f6ce312233e216252511bf7fdeef42d655526445edf62bb80b80cada69160482`.
All 14 admission artifacts are committed under
`results/sota/deform360_fresh_source_lock_v1/admissions`.

## Deterministic cohort rule

The lock:

1. verifies every admission and exclusion checksum;
2. discards rejected and excluded cases;
3. retains the lowest admitted episode per physical object;
4. checks that one object has one category label;
5. selects objects by category-sorted round robin and object-ID order.

The final object, episode, category, admission digest, method commit, method
configuration digest, and evaluator contract digest are sealed together.

The lock audits the evaluator contract itself. If public parity is incomplete,
its allowed claim is exactly:

```text
fresh_object_candidate_conventions_only
```

No caller-supplied flag can elevate that label.

## Locked cohort

The ordered 12-object cohort is now immutable in
`results/sota/deform360_fresh_source_lock_v1/deform360_fresh_object_cohort_lock_v1.json`,
internal SHA-256
`bafe26848ee83d8a4201e9d11d51af106370647f76ec702003e9ec51d3843729`.
It contains three filament, five sheet, and four volumetric objects. The lock
binds all five independent exclusion manifests, the unchanged method commit,
the source-lock configuration, and per-episode parity-contract SHA-256
`a8e903060ac58bfabd13ab1b43f15296fd685a04294f45a1c6c3c022280c1f95`.

The parity contract remains incomplete, so this cohort can establish only
fresh-object transfer under the fixed candidate metric conventions. No object
may be replaced after this lock.

## CLI

Admit one source episode:

```bash
bpt-prepare-deform360-fresh-source admit \
  /data/processed/OBJECT/episode_0000 \
  /data/raw/OBJECT/metadata.json \
  admission.json \
  --object-id OBJECT \
  --episode-id 0 \
  --category cloth
```

Build a target-free exclusion manifest:

```bash
bpt-prepare-deform360-fresh-source exclude \
  exclusion.json \
  --owner independent-held-owner \
  --object-id OBJECT_A \
  --object-id OBJECT_B \
  --source-sha256 SHA256_OF_OWNER_LOCK
```

Lock a cohort after all source admissions exist:

```bash
bpt-prepare-deform360-fresh-source lock \
  cohort-lock.json \
  --admission admission-a.json \
  --admission admission-b.json \
  --exclusion exclusion.json \
  --cohort-size 12 \
  --method-commit e2f8d827bfd60df79eeffee511a5df7e2d53ea21 \
  --method-config-sha256 METHOD_CONFIG_SHA256 \
  --parity-contract parity-contract.json
```

The next operational step is to seal every baseline and candidate prediction,
or an explicit technical-failure disposition, for all 12 locked cases. No
outcome-opening command is authorized until the all-case completeness barrier
passes.

## Real source-only smoke

The exact implementation commit
`ee9c93edcef8a7ac7631f12c4c201977793f7cde` was deployed in a clean checkout
on `gpuserver6000` and exercised on the already-open source case
`081-stripe-rope-ep0003`.

The admission passed with 12 cameras, 654 frame-zero points, 76 active rows,
and the released 60/16 train/test split. Its canonical source directory is
`081-stripe-rope`; the public `metadata.object` value is the descriptive label
`081-stripe`. The emitted artifact is
`results/sota/deform360_fresh_source_lock_v1/081-stripe-rope-ep0003.admission.json`,
file SHA-256
`f8849ae8d659421b90cde0202c2f6f398c370964b544a236b9173e63c381783a`.
Its internal admission digest is
`6e6e31e39b8dd8d28f9bccce2ccb4a85c34dded385ddb98b32103ea7c99045ec`.

This is an operational smoke only. The object was already open, no prediction
was run, and the result supplies no accuracy, calibration, prospective, or
SOTA evidence.

The complete repository suite on the unchanged implementation passed 775
tests with 28 expected skips in 337.02 seconds.
