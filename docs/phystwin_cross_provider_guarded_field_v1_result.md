# Cross-Provider Guarded Field V1 Result

## Decision

The frozen one-case prefix gate failed. The candidate therefore returned the
selected physical baseline byte-for-byte, the future-score artifact was not
opened after staging, and no larger source panel is justified.

This closes the tested composition:

```text
TAPNext++/CoTracker3 sparse causal witness
+ dense CoTracker3 endpoint displacement
+ relative-gauge correction
+ rank-4 persistent graph readout field
```

It does not close sparse causal observations, bias-aware online belief updates,
or nonpersistent dense updates generally.

## Frozen Prefix Result

The method was locked at implementation commit `d235b45` and protocol commit
`67c7a1c`. The prefix guard used manual identities 1 and 5 on frames
`[88,121)`, disjoint from provider identities 3, 4, 6, and 8 and future-score
identities 0, 2, and 7.

| Prefix predictor | Vector RMSE | Change from physical |
| --- | ---: | ---: |
| Selected physical baseline | 46.812 mm | -- |
| Sparse-provider graph field | 45.364 mm | -3.09% |
| Dense cross-provider field | 47.775 mm | +2.06% |
| Released candidate after guard | exactly physical | 0.00% |

The dense candidate was 5.32% worse than the sparse comparator. It therefore
failed both locked relative-gain requirements and both 0.25 mm absolute-gain
requirements.

## Diagnostics

- Dense endpoint support was 1,511 of 7,793 nodes (19.39%), above the locked
  10% minimum.
- Three sparse provider identities supported both the endpoint comparator and
  relative-gauge estimate.
- The relative tracker displacement bias was approximately
  `(1.13, -2.30, 0.92)` mm, with 1.55 mm residual radial RMSE.
- The robust dense projection downweighted 56.0% of accepted rows.
- Its weighted residual RMSE was 35.11 mm.
- The uncapped rank-4 field reached 65.89 mm, so the locked 10 mm cap was
  active.

This is not a support-starvation failure. The accepted dense block was broad,
but its displacement did not transfer to independent material identities.
Sparse provider evidence alone helped modestly on the prefix, while using it
only to gauge a dense persistent field made the prediction worse.

## Information Boundary

Prediction used the frozen selected physical trajectory, CoTracker3 frames
`[68,88)`, the sealed sparse provider, the rank-4 graph basis, and the staged
prefix-validation artifact. It did not read the staged future object points or
future manual identities. The prediction was then sealed.

Because the prefix gate rejected the method, the post-seal future scorer was
not run. No held-v8 or PokeFlex artifact was accessed.

## Evidence

The compact immutable evidence is in
`results/sota/phystwin_cross_provider_guarded_field_v1/`.

- Protocol SHA-256:
  `6ac6da132a8b5526456d6c24eb0fd2a8291bec0178d27b77bc3fca5b2ef2fb88`
- Prediction archive SHA-256:
  `1a02614d558700d790acf507014ef96d0a5f7e0d92b4723dafb00f4eb608b57d`
- Prediction report result SHA-256:
  `51ac60e6fb505377205bfff66c01b0c92b953d4f1ce9bbf9dcf6f42cbb6017e3`
- Prediction seal result SHA-256:
  `1ac3bf05bfca57db1816bc6013bb786b580399f1ef4d2f648cc1775cb88c6591`

The 57 MB prediction archive remains on `gpuserver4090` at
`/home/florianpfaff/bpt-cross-provider-field-v1-artifacts/prediction/prediction.npz`;
the repository records its digest rather than duplicating it.

## Verification

The exact archived implementation and protocol passed:

- 7 focused guarded-field tests;
- the complete server suite: 1,296 passed and 1 skipped in 11.83 seconds;
- Ruff on every changed Python file; and
- `git diff --check`.

The full-suite environment used the installed package and ImageIO's FFmpeg
7.0.2 binary. The server's system FFmpeg is too old to support the cadence
tests' locked `-fps_mode` argument.

## Recommendation

Do not expand this arm to the opened 22-case cohort and do not tune its rank,
cap, window, identities, or reliability thresholds on this case. The next
credible route must use more than a persistent camera-derived displacement
field. It should update a baseline-relative belief only when physical/action
support and an independently competent observation agree, retaining exact
fallback otherwise.
