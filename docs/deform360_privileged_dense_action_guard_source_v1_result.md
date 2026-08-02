# Deform360 Privileged Dense-Action Guard Source Result V1

## Decision

This post-open capacity control identifies a promising information pattern, but
it is not a deployable method and cannot support a state-of-the-art claim.

The control admits a pairwise-consensus RBF update only when:

1. applying its proposed displacement improves current dense material-state
   Chamfer by more than `0.1 mm`; and
2. the displacement has positive cosine with the known-action physical
   continuation.

On the opened Open27 source cohort, it admits 22 of 81 update intervals. Every
admitted interval improves both hidden-future identity RMSE and symmetric
Chamfer. On the already-open low-motion stress cohort, it admits none of 36
intervals and reproduces the selected physical/persistence backbone exactly.

The result says that trustworthy current material identity plus
action-consistent transfer could retain most of the source gain while
protecting near-static cases. It does not show that cameras can supply that
identity reliably.

## Why The Control Is Privileged

The current-shape test reads the score-family material identities at the update
frame. The forecast itself still uses no future object observation, and the
action-consistency term comes from the known-action physical rollout.
Nevertheless, observing at the update time the same material-identity family
that is later scored makes this an online-supervised capacity ceiling rather
than a causal camera-only predictor.

The `0.1 mm` gain threshold and positive-cosine rule were selected after the
Open27 and stress outcomes had been examined. They are therefore diagnostic
choices, not preregistered gates. No existing or fresh target cohort is
authorized under this rule.

## Opened-Source Result

Object-balanced hidden-future errors are:

| Arm | Identity RMSE | Symmetric Chamfer | Relative to selected |
| --- | ---: | ---: | ---: |
| Selected physical/persistence backbone | 8.807 mm | 7.888 mm | reference |
| Pairwise-consensus RBF candidate | 7.441 mm | 6.795 mm | -15.51% / -13.86% |
| Privileged dense-action guard | 7.658 mm | 6.946 mm | -13.05% / -11.94% |

The guarded control has 14 case-level joint wins, 13 exact ties, and no losses.
At the five-object level, four objects improve both metrics and one ties.
Mean object-level changes are `-1.149 mm` identity RMSE and `-0.942 mm`
Chamfer. These are descriptive opened-source results; five objects are not
enough to establish a portable regret guarantee.

Of the 81 source update intervals:

- 22 admit the candidate;
- all 22 admitted intervals improve both primary metrics; and
- 59 use the selected backbone exactly.

## Already-Open Stress Audit

The low-motion stress cohort is also post-open. Its selected backbone is much
stronger than the raw update:

| Arm | Identity RMSE | Symmetric Chamfer |
| --- | ---: | ---: |
| Selected physical/persistence backbone | 0.899 mm | 0.772 mm |
| Pairwise-consensus RBF candidate | 2.709 mm | 2.477 mm |
| Privileged dense-action guard | 0.899 mm | 0.772 mm |

All 36 stress intervals reject the update. The guarded trajectories are exact
fallbacks, so the aggregate is byte-for-byte equivalent to the selected
backbone.

This is useful contrastive evidence: pairwise camera consistency alone admits
coherent but harmful updates, whereas verified current material state
separates these opened examples. It remains a capacity result because that
verification is not independently observable in the deployed camera-only
setting.

## Scientific Consequence

The diagnostic sharpens the next-method target:

> Admit a discrepancy update only when an independently anchored causal
> material-state observation supports it and its transfer is consistent with
> the known action-conditioned physical response.

This is stricter than adding another camera-confidence threshold. Camera-only
uncertainty cannot distinguish true object translation from coherent
common-mode camera bias. The missing ingredient is an independent material
identity or modality, such as causally validated metric depth, tactile/contact
evidence, measured actuation, or a fresh source-validated equivalent.

The frozen V12 causal-response design expresses that target with disjoint
camera panels, earliest-event admission, physical/persistence selection,
nuisance removal, exact fallback, and synthetic positive/placebo controls. It
must not be evaluated until its independent all-attempt held-v8 exclusion
scope is available.

## Evidence

The machine-readable result is
`results/sota/diagnostics/deform360_privileged_dense_action_guard_source_v1/result.json`.
Its SHA-256 is
`4a7b26dcf662a8adbcee9bab8cf356a676de4aeb2f40c95328afb9a3a87801e1`.

The artifact contains source and stress interval decisions, current material
shape diagnostics, known-action context, all per-case scores, and explicit
flags that the observation is privileged and non-deployable. No held-v8 or
PokeFlex artifact was accessed.
