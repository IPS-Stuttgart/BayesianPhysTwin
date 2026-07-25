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

The next operational step is to run admission over a source-only candidate
pool and obtain exclusion manifests from every independent evaluation owner.
No outcome-opening command belongs in this package.

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
