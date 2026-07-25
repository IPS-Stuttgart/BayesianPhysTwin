# Deform360 fresh-object execution handoff v1

## What is ready

The frozen pairwise-consensus Bayesian update is ready for a genuinely fresh
public-object evaluation. Its method commit remains
`e2f8d827bfd60df79eeffee511a5df7e2d53ea21`; no post-open method change is
authorized.

The fail-closed source admission and cohort lock are implemented at
`a9d4737ce5e0b3113aefe1ed67f329dd14b88e42`. The exact implementation passed
the complete suite (774 passed, 28 skipped) and accepted one real, already-open
source bundle without deserializing future geometry. The smoke is operational
evidence only.

The machine-readable handoff is
`configs/sota/deform360_fresh_object_execution_handoff_v1.json`.

## Exclusion state

Two repository-owned manifests are ready:

| Scope | Objects | Internal digest |
|---|---:|---|
| Open-27 method development | 5 | `c8b79a1f6b76853229a5877428252ab69fcaf5b655f901e00b60cc1325795730` |
| Every public object named in tracked result artifacts/reports | 42 | `fe62c0f3284c078ecc44e9c1fce28fecbd17223a29bf2a95762efc5e85b0fcd2` |

The second set is deliberately conservative. Its plaintext inventory is
versioned because every listed identity was already public in this repository;
the exclusion artifact itself contains only namespaced hashes.

Independent hash-only manifests are still required from:

1. the held-v8 owner, covering every selected, reserved, opened, or technically
   dispositioned object across its attempts;
2. the frozen 34-object MolmoMotion-Field campaign owner, including the
   unsealable and retained technical-failure cases;
3. any other owner of an unpublished Deform360 cohort.

Those owners need only source/cohort identities. They must not inspect target
or score artifacts to create the exclusions.

## Fresh source custodian

After all exclusions arrive, an independent custodian should stage at least 18
source candidates so that source-contract failures do not force a post-lock
replacement. For each candidate, the custodian runs:

```bash
bpt-prepare-deform360-fresh-source admit \
  SOURCE_EPISODE_DIR RAW_OBJECT_METADATA OUTPUT_ADMISSION \
  --object-id OBJECT \
  --episode-id EPISODE \
  --category PREDECLARED_CATEGORY
```

The admission program may read raw object/episode identity, enum values,
source-stream and output hashes, camera names, split/contact counts, and the
frame-zero PLY header. It hashes `final_data.pkl` for custody but must never
deserialize it.

Rejected candidates remain recorded. Selection occurs only after every source
admission and exclusion artifact is immutable.

## Lock and custody

The coordinator then runs the `lock` command with every admission and every
independent exclusion manifest. The lock must contain 12 unique physical
objects, the unchanged method commit, source-lock configuration SHA-256
`fc8a68bd769a7b71d570e4c6879ea186b3ff4453457d29e4dc141a462a0038b7`,
and the current parity-contract digest.

After publication of the ordered lock:

1. prediction workers receive source/prefix data but no scoring target;
2. every object receives baseline and candidate prediction seals, or a
   registered technical-failure disposition;
3. a completeness barrier proves all 12 objects are accounted for;
4. one separately owned operator may then expose targets and score.

No object is replaced after lock.

## Claim boundary

The public Deform360 release still omits the benchmark evaluator and leaves
four authoritative fields unresolved: training cases, evaluation cases,
object aggregation, and missing-case policy.

Until those fields are supplied by released code or a content-hashed author
contract, the strongest permitted result is:

> Fresh-object transfer under explicit candidate metric conventions.

An official state-of-the-art claim additionally requires evaluator parity and
local reproduction of the strongest eligible baseline under that same
contract.

