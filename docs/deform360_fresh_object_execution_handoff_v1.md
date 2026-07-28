# Deform360 fresh-object execution handoff v1

## What is ready

The frozen pairwise-consensus Bayesian update is ready for a genuinely fresh
public-object evaluation. Its method commit remains
`e2f8d827bfd60df79eeffee511a5df7e2d53ea21`; no post-open method change is
authorized.

The fail-closed source admission and cohort lock are implemented at
`ee9c93edcef8a7ac7631f12c4c201977793f7cde`. It accepted one real,
already-open source bundle without deserializing future geometry, and the exact
implementation passed the complete suite (775 passed, 28 skipped). The smoke
is operational evidence only.

The machine-readable handoff is
`configs/sota/deform360_fresh_object_execution_handoff_v1.json`.

## Exclusion state

Five hash-only manifests are ready:

| Scope | Objects | Internal digest |
|---|---:|---|
| Open-27 method development | 5 | `c8b79a1f6b76853229a5877428252ab69fcaf5b655f901e00b60cc1325795730` |
| Every public object named in tracked result artifacts/reports | 42 | `fe62c0f3284c078ecc44e9c1fce28fecbd17223a29bf2a95762efc5e85b0fcd2` |
| MolmoMotion-Field frozen campaign (34 cases) | 17 | `18054955f5d8effb69eebc58aca2b3783e4e1fd0aa604f87bc2611f1f19a967c` |
| Prob4D opened/reserved/dispositioned objects (88 enumerated cases plus 23 object-only reservations) | 65 | `181796725382bcbe377b824dfac90243c6d3b0c9f9754fbeeb87cb6343d486ff` |
| Held-v8 published source-authorization history (23 case identities across 20 revisions) | 7 | `562640ce93bcb6c230dce8c684888e2895cb31e6a6f06b8e52858c263e667635` |

The repository-result set is deliberately conservative. Its plaintext
inventory is versioned because every listed identity was already public in this
repository. The independently supplied MolmoMotion-Field and Prob4D artifacts
contain only namespaced hashes and source-lock digests.

The held-v8 manifest was derived from all production-source revisions that
changed its authorization, lock-preparation, confirmation-source, or
replacement-source paths across the published v8, v8.1, v8.2, and v8.3
lineage. The audit read committed source only. It did not access the held
campaign root or any target, query, score, barrier, or outcome artifact, and
it emits no plaintext identities. All seven hashes were already contained in
the Prob4D exclusion, so the union count does not change.

No other externally managed cohort is presently known. A newly disclosed
cohort still requires a hash-only manifest before the final cohort lock; its
owner needs only source/cohort identities and must not inspect outcomes to
create it.

## Public source pool

The public Hugging Face directory catalog was snapshotted on 2026-07-26. It
currently exposes 190 object directories; the official README notes that the
public snapshot may be staged independently of the paper's reported 198
objects.

Prob4D subsumes all 42 repository-result hashes, and its 65 hashes have no
overlap with the 17 MolmoMotion-Field hashes. After applying their union of 82
physical-object exclusions, 108 public object identities remain. They are
recorded in
`results/sota/deform360_fresh_source_lock_v1/provisional_public_source_pool.json`.
The pool now incorporates the held-v8 manifest. It is ready for source
staging, but it is not a cohort lock.

A metadata-only preflight accepts 107 of these 108 identities. It rejects
`197-hand-sanitizer` because the manifest-bound public metadata contains a
`bimanual` value outside the exact `yes`/`no` domain. No camera, geometry,
tactile, future-position, or metric payload was read, and the rejected object
was neither normalized nor replaced.

## Fresh source custodian

The ordered 18-object source queue was frozen before any candidate payload was
inspected in
`configs/sota/deform360_fresh_source_staging_queue_v1.json`, internal SHA-256
`f80fed80ca2b9f1857539834bd92c6acb1b45a88eefbcae16e35cddaf9185d0e`.
It contains five filament, seven sheet, and six volumetric candidates, all at
episode zero. The morphology labels are source-only balancing strata, not
outcome labels.

An independent custodian should stage the candidates in frozen queue order so
that source-contract failures do not force a post-lock replacement. For each
candidate, the custodian runs:

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

Rejected candidates remain recorded. If fewer than 12 objects pass, processing
stops and a separately versioned reserve queue must be locked before any new
payload is inspected. Selection occurs only after every queued source
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
