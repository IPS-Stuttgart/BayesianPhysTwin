# TAPNext++ and CoTracker3 Complement V1

## Status

This is a post-open, one-case source-provider diagnostic. The method and gate
were frozen before the staged manual prefix was rescored. It neither changes
the failed TAPNext++ competence result nor authorizes simulator assimilation.

The frozen competence gate passed. This rescues the provider hypothesis, not a
Bayesian-PhysTwin accuracy claim.

## Method

The original TAPNext++ prediction remains authoritative wherever its frozen
two-view RGB-D lift has support. CoTracker3 is associated to each query at
frame 68 using only metric source geometry within 15 mm. Nearby dense tracks
form one correlated local displacement estimate; their count never increases
information mass and a 10 mm shared two-view bias floor remains in covariance.

CoTracker3 can fill a TAPNext++ hole only when all conditions hold:

1. at least three associated local tracks survive the unchanged two-view,
   3-pixel reprojection, and 0.1 quality checks;
2. the local displacement spread is at most 5 mm;
3. the same identity has at least five earlier TAPNext++/CoTracker3 overlaps;
4. prior cross-provider RMSE is at most 10 mm; and
5. the latest overlap is no more than five frames old.

The latest overlap is a causal gauge anchor. The filled row is

```text
TAPNext++ at the last common frame
+ CoTracker3 displacement since that frame.
```

An identity with no earlier TAPNext++ support receives exact fallback. There is
no fixed provider blend and no PhysTwin residual in prior perception
reliability.

## Gate

The provider must reach 75% supported prefix point-frames, preserve every
accepted TAPNext++ row bit-for-bit, keep total and endpoint RMSE below 15 mm,
and improve the newly added rows over exact persistence by at least 10%.

A pass permits only a separately locked disjoint-identity guarded-assimilation
study. It is not Bayesian-PhysTwin accuracy evidence, independent transfer, or
a state-of-the-art result.

The full immutable contract is
`configs/sota/phystwin_tapnextpp_cotracker_complement_v1.json`.

## Frozen Result

| Metric or gate | Result | Required | Pass |
| --- | ---: | ---: | :---: |
| Supported point-frames | 57/76 (75.00%) | at least 75% | yes |
| Total supported-row RMSE | 5.873 mm | at most 15 mm | yes |
| Endpoint RMSE | 8.218 mm | at most 15 mm | yes |
| Added-row RMSE | 11.126 mm | diagnostic | -- |
| Added-row persistence RMSE | 46.030 mm | comparator | -- |
| Added-row gain | 75.83% | at least 10% | yes |
| TAPNext++ rows bit-identical | yes | required | yes |

CoTracker3 recovered the five late rows of identity 6 after fourteen earlier
cross-provider overlaps. Identity 8 had no accepted TAPNext++ history and
therefore remained an exact fallback. This is the intended behavior: the
second provider repairs a causal loss of support, but cannot bootstrap trust
for an entirely unsupported identity.

The sealed evidence is under
`results/sota/phystwin_tapnextpp_cotracker_complement_v1/`. The evaluation
artifact has canonical SHA-256
`affc78056961b2e4ff866b15426b3fa4458d72f55ef8f8a3f5c8b59f464a8796`.

## Decision

A separately locked disjoint-identity guarded-assimilation experiment is now
justified. It must use new source cases or cross-fitting and must preserve the
same exact fallback. This one opened interaction cannot support an independent
transfer or state-of-the-art claim.
