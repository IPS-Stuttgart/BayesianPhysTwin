# Full-22 discrepancy candidate tournament v1

This experiment connects the merged source-only discrepancy tournament to the
released PhysTwin 22-case trajectory cohort. The cohort has already informed
method development. The result is therefore **retrospective source selection**,
not fresh independent validation.

The experiment may do exactly one of two things:

1. identify one candidate that passes both metric-specific source gates; or
2. retain `last_residual`.

It never authorizes a scientific claim, opens a still-sealed confirmation
payload, modifies the released PhysTwin trajectory, or promotes raw covariance
as calibrated deployment uncertainty.

## Frozen candidate roster

The protocol binds exact source revisions and configurations for:

- `physical_fallback`: the released PhysTwin trajectory without correction;
- `last_residual`: the last valid tracked residual, held deterministically;
- `independent_endpoint_v1`: the historical independent robust local-level
  endpoint model average;
- `dynamic_endpoint_v2`: persistence, local-level, and damped-trend model
  averaging with object-pooled causal evidence;
- `structured_kernel_rank4_v1`: the structured low-rank endpoint belief from
  PR #376; and
- `graph_dynamic_kernel_rank4_v1`: the graph-modal position/velocity belief
  from PR #377.

The structured and graph-modal candidates use the same rank-four basis. It is
constructed only from the released frame-zero tracked PhysTwin positions:

```text
K_ij = exp(-||x_i - x_j||^2 / (2 s^2)),
s = median nonzero pairwise distance.
```

A tiny frozen diagonal tie-break is added before the four largest kernel modes
are selected. Eigenvector signs are canonicalized. No residual, validation
outcome, future outcome, manual track, or confirmation value enters the basis.

## Nested information order

Each physical execution has three nested intervals:

```text
fit:              [0, fit_end)
guard validation: [fit_end, train_end)
scored future:    [train_end, frame_count)
```

The full experiment then replays selection while leaving out one complete
physical execution at a time. Frames, tracks, points, and views are never
treated as independent execution groups.

The executable pipeline has four separate publication stages.

### 1. Prefix preparation

`prepare-prefix` downloads the compact released files and publishes one NPZ per
case containing only:

- residuals and validity through `train_end`;
- the frame-zero geometry used by the common basis;
- the baseline, observations, visibility, and manual tracks through
  `train_end`, solely for the guard;
- the frozen tracked-to-full-state lift map; and
- exact split metadata.

No scored-future array is serialized in the prefix artifact.

### 2. Raw candidate prediction

`predict` is run separately for each candidate under its exact source revision.
For each case it:

1. fits through `fit_end` and seals validation forecasts;
2. refits the unchanged candidate through `train_end`;
3. seals every future forecast horizon without receiving the future outcomes;
4. records any technical failure explicitly; and
5. publishes a content-addressed prediction manifest.

A candidate failure produces zero raw forecasts and is rejected later. It does
not remove a case or candidate from the matched roster.

### 3. Prefix-only admission

`admit` reads only the prefix artifact and prediction manifests. It applies the
paper's metric-specific, baseline-relative regret guard to both primary metrics:

- official one-way L1 Chamfer distance; and
- official manual-track Euclidean error.

For each metric, it computes per-frame candidate-minus-baseline regret on the
validation interval and takes the frozen finite-sample `higher` 0.9 quantile.
The candidate is admitted for that execution only when both regret quantiles
are nonpositive. Technical failures and the registered physical fallback are
rejected explicitly.

All corrections are subject to the common 10 mm radial cap before evaluation.
Every rejection deploys the released PhysTwin trajectory exactly.

The resulting admission manifest is sealed before future scoring.

### 4. Future scoring and arbitration

`score` verifies the prefix, prediction, and admission identities before opening
the already-public future values.

It creates two complete tournament inputs:

- one with official manual-track error as point loss; and
- one with official Chamfer distance as point loss.

Each execution contributes early, middle, and late horizon units. Aggregation
and bootstrap resampling remain clustered by complete physical execution.

Both tournaments also report a Gaussian proper score on the tracked residual
field. A common 5 mm observation floor is added to every candidate covariance,
including deterministic methods. This score is a comparative source diagnostic;
it is not an interval-calibration claim.

Raw covariance intervals are deliberately disabled. The released full-22 audit
already established severe undercoverage, so apparent raw interval width or
coverage is not allowed to choose a candidate.

The final arbitration advances a challenger only when:

1. both metric-specific tournaments pass;
2. both select the same non-reference candidate;
3. no harmful accepted future unit is present;
4. mean proper score does not regress;
5. the paired point-loss bootstrap upper bound is nonpositive;
6. worst-group relative regression stays within the frozen bound; and
7. leave-one-execution-out selection is stable and held-execution
   nonregression passes.

Otherwise the final result is `last_residual`. The arbitration report marks
this valid negative result as `status="completed_no_selection"`, and the
workflow completes successfully after publishing the complete evidence.

Measured wall-clock runtime is retained in each prediction manifest, but the
runtime tie-break is disabled by supplying the same zero value to every
tournament candidate. This prevents machine scheduling from changing an exact
score tie.

## Reproduction

The GitHub workflow performs the complete information order and checks out
candidate implementations into separate exact-revision worktrees:

```bash
python scripts/science/run_full22_discrepancy_candidate_tournament_v1.py \
  prepare-prefix DATA_ROOT OUTPUT_ROOT/prefix \
  --protocol protocols/full22_discrepancy_candidate_tournament_v1.json

python scripts/science/run_full22_discrepancy_candidate_tournament_v1.py \
  predict OUTPUT_ROOT/prefix OUTPUT_ROOT/predictions/CANDIDATE \
  --protocol protocols/full22_discrepancy_candidate_tournament_v1.json \
  --candidate CANDIDATE \
  --source-revision REVISION

python scripts/science/run_full22_discrepancy_candidate_tournament_v1.py \
  admit OUTPUT_ROOT/prefix OUTPUT_ROOT/predictions OUTPUT_ROOT/admission \
  --protocol protocols/full22_discrepancy_candidate_tournament_v1.json

python scripts/science/run_full22_discrepancy_candidate_tournament_v1.py \
  score DATA_ROOT OUTPUT_ROOT/prefix OUTPUT_ROOT/predictions \
  OUTPUT_ROOT/admission OUTPUT_ROOT/result \
  --protocol protocols/full22_discrepancy_candidate_tournament_v1.json \
  --evaluator-revision EVALUATOR_REVISION
```

The final evidence contains:

- the prefix manifest;
- one prediction manifest and case NPZ set per candidate;
- the admission manifest;
- the complete raw scored rows;
- metric-specific tournament inputs and reports;
- the prediction-barrier record; and
- the final metric-arbitration report.

## Scientific boundary

A passing source tournament only selects a method for a separately frozen
future protocol. It does not establish:

- independent-object or independent-session transfer;
- calibrated raw covariance or predictive intervals;
- dynamically admissible simulator-state correction;
- Causal4D intervention benefit;
- deployment safety; or
- state of the art.

A negative result is complete. The opened full-22 cohort must not be retuned
afterward to rescue a rejected candidate.
