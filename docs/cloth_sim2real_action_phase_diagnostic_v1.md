# Cloth Sim2Real action-phase discrepancy diagnostic

## Evidence boundary

This is a post-open mechanism diagnostic on the nine Cloth Sim2Real dynamic
trials. All three repeat outcomes had already been opened by the frozen
online-belief study. For each evaluated trial, future observations from only
the other repeats of the same cloth and action train the phase profile. The
held repeat supplies its own causal prefix correction, but none of its future
observations enter fitting.

This is not an independent evaluation, an identical-information comparison
with published open-loop methods, or a state-of-the-art claim. It tests a
specific hypothesis suggested by the opened outcomes:

> The discrepancy relative to a causal prefix is action-phase dependent and
> repeatable across executions of the same action.

One linen trial rejected its prefix correction in the earlier frozen study and
therefore remains an exact fallback. Eight leave-one-repeat-out trials are
evaluable.

The compact result is
`results/sota/cloth_sim2real_action_phase_diagnostic_v1/summary.json`, with
SHA-256
`2b77e103c22d521bcf2077b02285482c08dc78126fd3fcff61c2a23dc2588336`.

## Diagnostic arms

All arms start from the same released physical rollout and the same prefix
correction accepted by the frozen online-belief method.

1. **Static persistence** holds the accepted spatial prefix correction fixed.
2. **Scalar phase** learns one normalized-horizon amplitude profile from
   disjoint repeats and applies it to the held prefix field.
3. **Translation delta** learns only three values per future frame: the robust
   coordinate-wise median residual change relative to each training repeat's
   prefix correction. It adds that profile to the held prefix field.

Profiles use fixed nine-frame smoothing. Scalar amplitudes are clipped to
`[-1.5, 1.5]`. Translation profiles and total node corrections retain the
existing 100 mm cap. A rejected prefix update returns the physical rollout
exactly.

## Leave-one-repeat-out result

Across the eight evaluable trials, scalar phase beats static persistence in
8/8 directed-Chamfer comparisons and 7/8 symmetric comparisons. Translation
delta beats persistence in 8/8 for both metrics.

| Comparison over 8 evaluable trials | Directed L1 CD | Symmetric L1 CD |
| --- | ---: | ---: |
| Scalar phase vs static persistence, ratio of means | 5.54% better | 1.85% better |
| Translation delta vs static persistence, ratio of means | **35.98% better** | **27.43% better** |

The leave-one-repeat-out scalar profiles correlate with the corresponding held
profiles from 0.905 to 0.991. This supports temporal repeatability within the
same cloth/action family.

## Opened target-repeat view

Repeat 2 had served as the independent target in the earlier frozen study. It
is reported here only as an opened diagnostic subset:

| Cloth | Physical directed L1 CD | Static persistence | Scalar phase | Translation delta |
| --- | ---: | ---: | ---: | ---: |
| Chequered rag | 64.20 mm | 58.59 mm | 54.78 mm | **41.33 mm** |
| Cotton rag | 80.99 mm | 76.44 mm | 71.43 mm | **45.58 mm** |
| Linen rag | 86.19 mm | 85.16 mm | 82.28 mm | **52.93 mm** |
| Object-balanced | 77.13 mm | 73.40 mm | 69.49 mm | **46.62 mm** |

On this opened subset, translation delta improves the ratio-of-means directed
metric by 36.49% relative to static persistence. Because the other repeats'
future outcomes train the profile, this number is hypothesis-generating only.

## Interpretation

The result rejects constant-in-time discrepancy as the best description of
these repeated dynamic actions. Much of the transferable error is global or
very low frequency and changes coherently with action phase. A suitable
Bayesian model would therefore separate:

```text
prefix-specific spatial correction
+ action/phase-conditioned shared discrepancy
+ execution-specific uncertainty
```

The current diagnostic does not establish transfer to a new action, object,
or action speed. It also provides only a mean correction and no calibrated
predictive covariance.

## Next gate

A confirmatory method must be frozen before observing outcomes from fresh
executions and must train the action-conditioned component only on source
actions. The decisive evaluation is a held-action or held-profile test with:

* an unchanged physical and static-persistence baseline;
* exact fallback when source support is insufficient;
* execution-clustered accuracy intervals;
* covariance that includes between-repeat profile variation;
* separate early, middle, and late metrics;
* no target-fitted phase alignment or amplitude.

Until that test exists, the frozen online-belief result remains the supported
Bayesian-PhysTwin claim and this diagnostic supplies the next model hypothesis.
