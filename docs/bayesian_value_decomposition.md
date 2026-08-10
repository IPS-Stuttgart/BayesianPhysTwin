# Bayesian-value decomposition

`bpt evidence decompose` attributes the result of one matched prospective study
to four predeclared comparison arms. It consumes the ordinary
`bayesian-phystwin-decisive-evidence-v1` contract, including output from
`bpt evidence score`.

## Required arms

The four roles are supplied explicitly on the command line:

1. **Deterministic reference**: last residual or another matched deterministic
   predictor, always deployed without a guard.
2. **Guarded reference**: the identical deterministic prediction with the
   source-frozen uncertainty and fallback policy.
3. **Bayesian mean**: the Bayesian predictive mean under the same guard,
   reliability, identifiable rank, and interval-width registration as arm 2.
4. **Full belief**: the complete recursive belief, including any mixture,
   structured covariance, reliability update, and resulting guard decision.

The analyzer rejects a study when the deterministic arm is not always deployed,
when arms 1 and 2 have different raw loss, or when arms 2 and 3 do not share the
same acceptance, risk score, reliability, identifiable rank, and registered
interval widths. These invariants make each adjacent comparison interpretable:

- arm 1 to arm 2: uncertainty and guard value;
- arm 2 to arm 3: Bayesian mean value under the common guard;
- arm 3 to arm 4: value of the full recursive distribution and its policy.

## Reported views

Each metric contains raw and deployed decompositions under two aggregations:

- one equal weight per registered unit;
- one equal weight per independent `group_id`, after averaging registered units
  within a group.

Each step reports candidate-minus-baseline loss, mean improvement with positive
values denoting benefit, relative change, and paired wins, ties, and losses. The
three steps must telescope exactly to the deterministic-to-full comparison.

A paired group-clustered bootstrap is also run for every step and the total. The
same deterministic seed and metric-derived resampling stream are used by the
existing decisive-evidence bootstrap implementation, so frames, points, tracks,
or repeated horizons cannot inflate the independent sample size.

## Command

```bash
bpt evidence decompose \
  runs/prospective/proper-score-evidence.json \
  runs/prospective/bayesian-value-decomposition.json \
  --deterministic-reference last_residual \
  --guarded-reference last_residual_guarded \
  --bayesian-mean bpt_mean \
  --full-belief bpt_full
```

The report binds the source evidence through a canonical content identity and is
published atomically with the exact input-file SHA-256. The four names are roles,
not hard-coded method labels, so a frozen experiment may use more specific names
without changing the analysis contract.

## Interpretation

A positive total does not imply that every component helped. For example, a
Bayesian mean can improve raw loss while a poorly calibrated guard removes that
benefit, or the guard can be valuable even when the Bayesian mean is tied with a
simple deterministic predictor. The adjacent steps and both raw/deployed views
make those cases visible instead of assigning all improvement to one label.

## Scientific boundary

The decomposition attributes evidence inside one fully matched registered study.
It does not make an implemented arm empirically valid, convert retrospective data
into confirmation, or establish calibrated uncertainty, independent transfer,
Causal4D benefit, deployment safety, or state of the art.
