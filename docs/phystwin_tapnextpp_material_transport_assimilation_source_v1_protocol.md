# TAPNext++ Material Transport Assimilation Source Protocol

## Question

The frozen provider study passed on 14/14 source cases with 99.19% support,
5.006 mm case-balanced identity RMSE, and a 74.18% gain over exact
persistence. This separately locked experiment asks:

> Does an accurate causal sparse observation improve the untouched future
> when it is transported through one immutable material-node identity?

The cases are already-open PhysTwin source data, but they are disjoint from
the failed eight-case source-geometry association panel. They can diagnose
the new bridge; they cannot independently confirm it.

## Frozen Material Bridge

For material identity `i`, let `g(i)` be its graph node fixed from the global
frame-zero geometry. Over the terminal causal prefix, form

```text
e[i,t] = (y[i,t] - y[i,0]) - (x[g(i),t] - x[g(i),0]).
```

The first provider row anchors the displacement and is not an update. A
constant query-to-node offset cancels, while the measured query-frame
attachment distance enters metric covariance. Node identity, association
probability, and prior reliability do not depend on the state innovation.
The innovation is processed once by the existing robust mixture likelihood.

The endpoint posterior is injected directly at the fixed material nodes and,
for the primary arm, propagated with the same graph-Laplacian posterior,
rank-independent 0.1 prior strength, 10 mm cap, and four-effective-row limit
used in the failed arm. No future result selects a cap, graph strength, arm,
or case.

## Arms and Metrics

The four arms remain:

1. released physical rollout;
2. fixed dense endpoint persistence;
3. dense persistence plus fixed-node direct update;
4. dense persistence plus fixed-node graph update, the primary arm.

Future scoring reports Chamfer distance, all manual identities, the four
queried identities, disjoint hidden identities, late horizon, conditional
90% coverage, and NEES.

## Advancement Gate

Relative to dense persistence, the primary graph arm must simultaneously:

- improve case-balanced Chamfer distance by at least 5%;
- improve all-identity track error by at least 5%;
- improve queried-identity track error by at least 10%;
- regress hidden-identity track error by no more than 2%;
- jointly avoid CD and all-track regression in at least 10 of 14 cases;
- retain at least 95% hidden future-frame support;
- attain at least 80% conditional 90% coverage and move no farther from 90%
  than dense persistence; and
- use exact dense fallback for any failed provider case.

Every case is retained. Failure stops this arm without tuning against the
opened futures. A complete pass authorizes only a new protocol on genuinely
fresh objects; held-v8 remains outside this experiment.

## Custody

The provider summary and provider source manifest are bound by file and
canonical hashes in the protocol. Staging verifies each sealed carrier and
immutable material attachment, then writes a prediction input and a separate
withheld future artifact. All 14 predictions must seal before any future
artifact is opened.

Manual benchmark identity initializes the provider query and fixes its graph
node. This is a source capacity study of material transport, not deployable
automatic association or an independent state-of-the-art claim.
