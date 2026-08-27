# PokeFlex source-to-later-interaction transfer calibration v1

## Question

The existing PokeFlex results show that a correction magnitude selected from two
opened interactions improves a third prospectively locked interaction and also
improves the retrospective public official13 subset. This analysis asks a
stronger mechanism question using only those already retained artifacts:

> Does the strength of the two-source-action signal rank the benefit obtained on
> later interactions of the same physical object?

A positive relationship is more informative than an aggregate panel gain. It
means that the source evidence is calibrated to later benefit rather than merely
selecting a globally favorable correction.

## Frozen inputs and statistical unit

The analysis consumes only three committed, content-addressed artifacts:

- `configs/sota/pokeflex_action_robust_scale_v3.json`;
- `results/sota/pokeflex_action_robust_fresh6_v3/summary.json`; and
- `results/sota/pokeflex_action_robust_official13_public_v1/summary.json`.

The source statistic is each object's mean relative improvement over the global
scale across its two opened source interactions. The target statistic is the
equal-interaction mean robust-over-global relative gain across every retained
later interaction. Objects occurring in both target panels are averaged before
inference, leaving **11 physical objects** as independent units and **14 later
interactions** as nested observations. Six interactions came from the
prospectively locked fresh6 panel; eight came from the retrospective official13
overlap.

## Exact result

Source strength strongly ranks later benefit:

```text
midrank Spearman rho = 0.9178082191780822
exact one-sided permutation tail = 3732 / 39916800
p = 9.349446849446849e-05
```

The permutation probability is exact, not asymptotic or Monte Carlo. A subset
dynamic program counts all `11!` labelled assignments while preserving average
ranks for ties.

The result is not driven by one object. Omitting each physical object in turn
gives Spearman correlations from `0.8902439024390244` to
`0.9512195121951219`. Replacing each object's mean later gain by its **worst**
retained later-interaction gain still gives

```text
rho = 0.7442922374429224
exact p = 221350 / 39916800 = 0.005545284191117525
```

Nine objects improve over the global correction, none regress, and two tie. The
one-sided exact sign probability among the nine non-ties is `1/512 =
0.001953125`.

## Exposure-stratified interpretation

The six fresh6 interactions remain the only prospective target panel for the
repeated-action scale rule. On those six objects alone, the rank statistic is
positive but underpowered (`rho=0.5428571428571428`, exact one-sided
`p=107/720=0.1486111111111111`). The eight-object official13 overlap is
retrospective (`rho=0.6867469879518072`, exact one-sided
`p=1374/40320=0.03407738095238095`).

The combined 11-object result is therefore a **retrospective cross-panel
mechanism diagnostic**, not a newly prospective experiment. Its contribution is
the source-to-target calibration pattern: stronger repeated-action evidence is
associated with larger later-interaction benefit, and the relationship survives
worst-interaction and leave-one-object-out analyses.

## Claim boundary

This analysis uses no new acquisition and opens no new target. It supports the
bounded statement that the frozen source-action evidence strength ranks later
benefit on previously studied PokeFlex objects. It does not establish
unseen-object transfer, prospective confirmation of the rank statistic, a unique
physical-state interpretation, full official-split reproduction, deployment
safety, or state of the art.

The complete retained result is
`results/analysis/pokeflex_source_target_transfer_v1/result.json` with artifact
ID `089397f1a3cc8ca393f6e38bf0963045ddca7218b3c11307a8f3f2b4217e8bd4`.
