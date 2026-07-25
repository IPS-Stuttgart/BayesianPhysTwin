# Deform360 Selective Bias-Aware Guard: Post-Open V1

## Status

This is a **post-open mechanism diagnostic** on the exhausted 23-case
selective-camera cohort. It is not prospective confirmation, selector tuning,
or a new accuracy claim. The source-v4 lock was applied unchanged, and each
candidate was constructed without a target before its already-open score was
joined.

## Question

The sealed camera-only update regressed catastrophically against persistence:

| Metric | Persistence | Sealed camera update | Relative regression |
| --- | ---: | ---: | ---: |
| Hidden identity RMSE | 0.384 mm | 6.780 mm | +1665.49% |
| Hidden symmetric Chamfer | 0.214 mm | 6.377 mm | +2873.34% |

The diagnostic asks whether the already frozen bias-aware source-v4 rule would
admit these updates or preserve the unchanged baseline. It does not ask whether
v4 improves this cohort; the action-only windows make persistence nearly
unbeatable.

## Frozen Input Path

For every case, the adapter verifies the pre-outcome hashes of:

- the sparse RGB-prefix measurement;
- the selective-camera prediction;
- the selective-camera prediction report; and
- the frame-zero physical backbone.

It derives the target-free physical response as

```text
sealed driven backbone - sealed persistence
```

and derives action support only from nonzero response nodes. Reliability uses
triangulation redundancy and reprojection error, never the state innovation.
The selective archives do not contain cycle covariance, so the recorded v4
variance floor is supplied. In this cohort, that variance is never consumed by
an accepted update because physical-response eligibility is evaluated first.

## Result

All 23 sealed physical backbones are bit-exact persistence. Consequently:

| Quantity | Result |
| --- | ---: |
| Cases / objects | 23 / 12 |
| Update intervals | 69 |
| Candidate-available intervals | 0 |
| Accepted intervals | 0 |
| Byte-exact fallback intervals | 69 |
| Guarded trajectory differs from persistence | 0 cases |

The guarded result is therefore exactly 0.384 mm identity RMSE and 0.214 mm
Chamfer, a 0.00% change from persistence. This avoids every sealed-camera
regression by abstaining; it provides no accuracy gain.

## Interpretation

This is a useful safety result and a sharp design constraint:

1. Camera-internal consistency is insufficient under coherent common-mode
   bias.
2. Physical/action support blocks unsupported virtual-sensing updates.
3. A bit-exact fallback makes rejection auditable rather than approximate.
4. No observation update can improve these windows without an informative
   baseline or independent evidence, because the sealed physical backbone has
   zero response and the true motion is usually below 1 mm.

The result does **not** validate the v4 update on fresh objects. It also does not
justify changing the opened selective selector. A future accuracy experiment
must preselect windows with both causal observed object response and nonzero
physical response, while keeping the outcome and future RGB sealed.

## Evidence

- `results/sota/deform360_selective_bias_aware_postopen_v1/result.json`
- `results/sota/deform360_bias_aware_guarded_belief_v4/prospective_lock.json`
- `scripts/remote/diagnose_deform360_selective_bias_guard.py`
- `src/bayesian_phystwin/deform360_selective_bias_guard_diagnostic.py`

The remote run directory is
`/mnt/corsair/florianpfaff/bpt-selective-bias-guard-diagnostic-v1` on
`gpuserver6000`. The result artifact has canonical SHA-256
`f106265522a25f4ce11df68283b61594e3b5d51e7b31788398d9e56b8fac9128`.
