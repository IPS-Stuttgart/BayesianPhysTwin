# Dynamic TAPNext++ Budgeted Hidden-Transfer Audit

## Status

This is a post-open diagnostic on the one V3 source case with valid
birth-to-update support. It is not method-selection, transfer, confirmation,
calibration, or state-of-the-art evidence. Fresh objects, the V1 sealed target
cohort, held-v8 artifacts, and future camera observations remained unopened.

The target-free subset-selection code was frozen at
`27abdbe5ab2cfb668641b90f28433df23b774820` before this audit read the
already-open source target.

## Question

The V3 hidden score excluded all 46 motion-targeted query identities. This
left an almost static hidden set and could have hidden useful transfer to
unobserved active identities. The audit therefore:

1. retained deterministic farthest-point subsets of 1, 2, 4, or 8 supported
   identities using frame-zero physical geometry only;
2. reran the frozen V3 set-valued assimilation;
3. excluded only the retained subset from the future score, leaving all other
   query identities hidden;
4. computed a target-using per-identity physical-versus-persistence oracle as
   a diagnostic ceiling.

## Result

| Budget | Persistence identity RMSE | Candidate identity RMSE | Physical identity RMSE |
| ---: | ---: | ---: | ---: |
| 1 | **0.089 mm** | **0.089 mm** | 10.965 mm |
| 2 | **0.089 mm** | **0.089 mm** | 10.965 mm |
| 4 | **0.089 mm** | 6.538 mm | 10.960 mm |
| 8 | **0.090 mm** | 6.523 mm | 10.948 mm |

Budgets one and two correctly selected exact persistence because they did not
meet the frozen selector-support requirement. Budgets four and eight selected
the physical backbone from the sparse observations and regressed by more than
seventyfold in hidden identity RMSE. Their pairwise correction gate also
rejected because fewer than the frozen nine inliers were available, so the
candidate remained the selected physical continuation.

The diagnostic per-identity oracle selected persistence for every hidden
identity:

| Budget | Hidden physical choices | Hidden persistence choices |
| ---: | ---: | ---: |
| 1 | 0 | 451 |
| 2 | 0 | 450 |
| 4 | 0 | 448 |
| 8 | 0 | 444 |

This includes the unretained motion-targeted query identities. Persistence
also dominated in the near, middle, and far distance bands.

## Interpretation

The V3 failure was not merely caused by removing every high-motion query from
the hidden score. On this supported source case, the automatic provider's
motion evidence did not identify a region where either the physical
continuation or its current RBF correction beat exact persistence on unseen
material identities.

The result therefore removes the empirical premise for implementing a V4
compact-support decoder from this provider: even an oracle local
physical-versus-persistence field has zero hidden physical choices here.
Changing the kernel can limit damage, but it cannot create predictive
headroom.

This does not establish that TAPNext++ is generally unsuitable. It establishes
that the current causal multiview material-association provider has no
transferable hidden-identity headroom on its only supported V3 source case.
A new attempt would first need an independent source competence result in
which automatic sparse identities beat persistence on disjoint hidden
identities, not merely on their own observed endpoints.

## Decision

Do not implement or evaluate the proposed V4 local-mixture method on a fresh
cohort. Preserve exact fallback and move SOTA effort to the separately
source-proven baseline-relative guarded online-belief route. Revisit a
TAPNext++ provider only if a different query/association mechanism passes a
disjoint hidden-identity competence gate on multiple source objects.

## Artifacts

- Audit:
  `results/sota/diagnostics/deform360_dynamic_tapnextpp_budgeted_v4/source_budget_audit.json`
- Canonical result SHA-256:
  `cfdd167490e4e159f50e03223ee2c90489a040cb128b6a9419d2d7b9188189f4`
- File SHA-256:
  `b46d393561f73529310cd3dd4283e5f2fcd8bb23f45d25aaad648ef560c978db`
