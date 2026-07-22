# Deform360 lifecycle-equivalence development incident

Date: 2026-07-22 CEST

Status: development-only protocol incident. No formal held numerical payload
was intentionally opened or deserialized, and no formal qualification or
scientific outcome is affected. The broad discovery traversal may have
accessed formal-held path metadata.

## Boundary that was crossed

The repeated-fit outputs at

```text
/mnt/corsair/florianpfaff/bpt-resource-lifecycle-ab-variability-0db75dc3a54cba682e5398caac4d301b96aa412f
```

were intended to remain numerically unopened until the equivalence analyzer
and its acceptance rule had been committed. An independent security reviewer
mistakenly treated its review assignment as authorization to validate those
outputs. Exact start and end times were not recorded: the early accesses had
completed by 22:44:53 CEST, and the final recomputation completed before
22:53:30 CEST. It read the ten development PLY files and their fit evidence
and performed a 2.7-second CPU recomputation of the twelve non-render pairwise
metrics.

The reviewer reported that all twelve non-render metrics passed the already
written empirical envelope. Disclosed examples included:

- cross-mode median XYZ-mean distance `2.426e-5 m`, against a `3.092e-5 m`
  within-mode limit;
- cross-mode p95 XYZ-p95 distance `4.489e-5 m`, against a `4.509e-5 m`
  within-mode limit;
- cross-mode p95 relative-count difference `6.036e-4`, against a `7.243e-4`
  within-mode limit; and
- cross-mode p95 quaternion-angle difference `0.02034 rad`, against a
  `0.02061 rad` within-mode limit.

It also disclosed vertex counts of `8281, 8284, 8281, 8283, 8282` for the
original fits and `8279, 8284, 8278, 8284, 8281` for the wrapped fits. It
confirmed that the inert exported normal fields were zero and that all fits
used physical GPU 0.

No gsplat render was run. No RGB, alpha, PSNR, SSIM, LPIPS, or other
image-space metric was computed or inspected. No analyzer result or repeat
manifest was found or read. The remote commands intentionally created no
artifact, log, render, manifest, or result; possible access-time metadata
changes were not measured. Local review commands may have refreshed ignored
Python bytecode caches.

## Pre-disclosure contract and permitted changes

Before the incident, the stopped analyzer candidate and tests had been
reported with these file hashes:

```text
analyzer  5cd52db010806212e9254c69462a93f956b8f719bb2f254f09476138b9b7b09d
tests     231d96409a49cf859861d1c8e10a87fe34f45a3562359120ac50db55dcd4f9cd
```

That candidate already fixed the metric names, pair construction, repeat
minimum, linear quantiles, and the two per-metric inequalities:

```text
cross median <= max(within-original p95, within-wrapped p95)
cross p95    <= max(within-original max, within-wrapped max)
```

All metrics must pass. These statistical choices, the exact-or-secondary
acceptance rule, and all numerical thresholds are frozen and must not change
in response to the disclosed values. Subsequent edits are limited to findings
already raised by the independent code review: exact provenance and schema
validation, source-dataset and GPU binding, process isolation, pre/post
revalidation, pinned-exporter float32/zero-normal invariants, crash-safe
publication, explicit generator profiles, and a nonzero scientific-NO-GO exit
code.

## Consequence and remediation

The existing GPU-0 diagnostic must be described as **partially unblinded and
not commit-preregistered**. It may be used only as development evidence that
the resource wrapper is plausible to qualify; it is not a paper result and is
not formal admission evidence.

The formal lifecycle qualification is a new v2 protocol. It will generate a
fresh five-original/five-wrapped cohort on physical GPU 1 from one clean H1
generator/analyzer tree, apply the unchanged analyzer, and then run the
243-fit resource soak. Its complete evidence closure will be created inside a
new self-contained qualification root. No v1 or GPU-0 output can satisfy the
v2 protocol. The subsequent held attempt also uses fresh predictions,
reconstructions, queries, and scores.

The reviewer was stopped as soon as the disclosure was reported. It stated
that it crossed the boundary because it overlooked the explicit
no-output-inspection instruction; there was no technical need to do so.
