# Deform360 Pairwise Regret Guard Source Result V1

## Decision

The source-trained baseline-relative regret certificate is safe but vacuous.
It accepts zero of 81 object-held-out Open27 update intervals and zero of 36
intervals in the already-open low-motion stress cohort. Every rejection is an
exact selected-backbone fallback.

Do not promote this certificate as the guard for the pairwise-consensus RBF
update. The underlying update retains its strong opened-source result, but a
nontrivial 90% object-level upper regret bound cannot be estimated from only
five independent source objects.

## Candidate And Baseline

The target-free candidate preserves two separate physical and persistence
belief states. At each update it:

1. selects the lower-current-Chamfer backbone using only the permitted
   observation;
2. applies the frozen pairwise-consensus association gate;
3. updates the selected recursive RBF belief once; and
4. falls back byte-for-byte to the selected backbone when association is
   rejected.

The implementation reproduces the previously frozen
`raw_selected_backbone_full_blend_rbf_pairwise_clique` trajectory exactly.
The transferred Open27 bundle is bound by manifest SHA-256
`02babb4e041fce354a168a76c055a043032dbbf5eb5320e2c8a0e409bb2d83bb`.

## Opened-Source Result

Object-balanced hidden-future errors are:

| Arm | Identity RMSE | Symmetric Chamfer |
| --- | ---: | ---: |
| Selected physical/persistence backbone | 8.807 mm | 7.888 mm |
| Pairwise-consensus RBF candidate | 7.441 mm | 6.795 mm |
| Object-cross-fitted regret guard | 8.807 mm | 7.888 mm |

The unguarded candidate improves the selected backbone by 15.51% identity
RMSE and 13.86% Chamfer. The regret guard gives up all of that gain because it
accepts no held-object interval.

## Why The Certificate Is Vacuous

The certificate is fitted by leaving out one physical object at a time. With
five source object groups, the finite-sample 90% upper residual quantile uses
rank 5 of 5 and has only `5/6 = 83.33%` finite-sample resolution. The worst
held-object residual therefore controls the bound.

The full-source upper residual quantile is `8.390 mm`, larger than the
predicted gain of every interval. Of the 81 cross-fitted decisions:

- 58 are finite-bound exact fallbacks;
- 23 are outside-source-support exact fallbacks; and
- 0 accept the candidate.

Adding causal pre-update belief features does not repair this sample-size
limit. The strongest new diagnostic, innovation RMS normalized by object
scale, has a direction-free interval-win AUC of 0.816 on the opened source.
It is useful evidence that updates need physical-response support, but it is
not a valid substitute for an object-level regret bound.

## Already-Open Stress Audit

The optional stress cohort is post-open and cannot support confirmation. It
contains action-only windows for which persistence is already exceptionally
strong:

| Arm | Identity RMSE | Symmetric Chamfer |
| --- | ---: | ---: |
| Selected physical/persistence backbone | 0.899 mm | 0.772 mm |
| Pairwise-consensus RBF candidate | 2.709 mm | 2.477 mm |
| Source-trained regret guard | 0.899 mm | 0.772 mm |

The unguarded update regresses by 201.34% identity RMSE and 220.84% Chamfer.
The formal certificate accepts zero intervals and restores the baseline
exactly. This demonstrates safe fallback, not useful selectivity.

The stress result also confirms the identifiability limit behind the guard:
camera-coherent motion can have strong pairwise consistency and large apparent
innovation while still being common-mode bias. A camera-only magnitude,
uncertainty, or temporal-consistency threshold cannot certify improvement
under arbitrary coherent bias.

## Consequence

The next credible evaluation needs at least one of:

- substantially more independent source objects for a nonvacuous
  baseline-relative regret bound;
- a separately validated observation modality that can identify common-mode
  camera bias; or
- a genuinely fresh dynamic-object cohort on which a source-frozen
  physical/action-supported admission rule can be evaluated.

Do not weaken the confidence level, treat episodes from the same object as
independent groups, or tune a threshold on the opened stress outcomes merely
to obtain nonzero admissions.

## Evidence

The machine-readable result is
`results/sota/diagnostics/deform360_pairwise_regret_guard_source_v1/result.json`.
Its SHA-256 is
`30b8659c7d965d47257ad05dc8a79164138c3cb6c850a3ba7c3862a432845d18`.

The result contains all interval features, cross-fitted decisions, the
full-source certificate, aggregate metrics, and the explicitly labeled
post-open stress audit. No held-v8 or PokeFlex artifact was accessed.
