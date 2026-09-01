# DLO3 cross-backend residual geometry with one scalar

## Why this experiment exists

Exact no-refit coefficient transfer is the strongest cross-backend test: the
DEFORM-fitted local-residual models are applied unchanged to sealed PyElastica
predictions. A failure of that point-prediction gate, however, does not
necessarily imply that the learned discrepancy is entirely backend-specific.
The high-dimensional residual field may point in a shared physical direction
while the two simulators express that missing effect at different amplitudes.

This registered fallback separates **geometry** from **amplitude**. The complete
DEFORM residual field remains fixed. Only one scalar multiplier is estimated,
and the trajectory being evaluated is excluded from the scalar fit.

## Frozen operator

Let `b_i` be the raw PyElastica prediction, `p_i` the equal-seed direct DEFORM
transfer prediction, and `y_i` the observed trajectory for complete trajectory
`i`. The transferred correction and true PyElastica residual are

```text
c_i = p_i - b_i
r_i = y_i - b_i.
```

For each held-out trajectory `i`, the scalar is fitted on the other seven
complete trajectories:

```text
alpha_-i = clip(
    sum_{j != i} <c_j, r_j> / sum_{j != i} <c_j, c_j>,
    0,
    4,
)
```

The held-out prediction is

```text
p_scalar_i = b_i + alpha_-i c_i.
```

There is no intercept, no node-specific parameter, no coordinate-specific
parameter, no feature update, and no refitting of the DEFORM residual field.
The scalar is selected by coordinate L2 on seven trajectories; the held-out
score remains mean coordinate L1 over all nodes and frames.

## Registered evidence gates

Shared residual geometry is supported only when both gates pass.

The cross-validated point gate requires:

- at least 1% mean L1 improvement over raw PyElastica;
- at least 6 of 8 held-out trajectory wins; and
- no held-out trajectory ratio above 1.10.

The directional gate requires:

- positive transferred/true-residual cosine on at least 6 of 8 trajectories;
- median trajectory cosine of at least 0.05; and
- a positive fitted scalar in at least 6 of 8 folds.

The exact no-refit arm and the PyElastica-specific high-dimensional refit remain
visible comparators. The registered output reports a claim ladder rather than
collapsing these distinct levels of transfer.

## Information boundary

This is a retrospective source-only diagnostic. The eight DLO3 source-test
outcomes were opened in predecessor studies, and each fold uses the other seven
source-test outcomes to estimate its scalar. The held-out trajectory's label is
never used for its own scalar.

The following remain forbidden:

- any high-dimensional PyElastica refit;
- any DEFORM coefficient update;
- DLO3 official evaluation;
- DLO4 or DLO5 payloads;
- held-v8 data; and
- paper-claim authorization by the evaluator itself.

The paired bootstrap interval is descriptive for the fixed cross-validated
predictions; it does not account for dependence induced by overlapping
seven-trajectory training folds and is not part of the promotion decision.

## Claim ladder

A successful exact-transfer gate supports **no-refit coefficient-level
cross-backend transfer**.

If exact transfer fails but the registered scalar and directional gates pass,
the bounded conclusion is instead:

> The high-dimensional DEFORM discrepancy field exhibits residual geometry
> shared with PyElastica, while backend amplitude requires one-dimensional
> recalibration.

That remains materially stronger than demonstrating only that two simulators
can each be corrected after separate high-dimensional fitting. It is weaker than
exact no-refit transfer and does not establish arbitrary-backend transfer,
zero-shot object generalization, fresh target confirmation, safety, or state of
the art.

## Execution priority

This fallback uses the same repository-wide concurrency group as the exact
no-refit experiment. Neither can claim `gpuserver4090` until protected DLO4/DLO5
run `33361441865` is terminal, and they cannot execute concurrently with one
another.
