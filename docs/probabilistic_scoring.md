# Proper scoring for registered physical-query predictions

`bayesian_phystwin.probabilistic_scoring` evaluates matched predictive
**distributions** in a caller-frozen physical-query space. It complements the
existing decisive-evidence analysis, which evaluates deployed loss, exact
fallback, harmful accepted updates, risk--coverage, and interval calibration.

The scorer is diagnostic infrastructure. It does not turn a readout-discrepancy
belief into a latent physical-state posterior and does not authorize a
calibration, transfer, intervention, safety, or state-of-the-art claim.

## Supported scores

The version-1 contract supports four lower-is-better scores:

- `energy_score`: the weighted empirical multivariate energy score. Pairwise
  distances are accumulated in bounded blocks rather than in one complete
  sample-by-sample matrix.
- `variogram_score`: a pair-weight-normalized empirical variogram score over a
  predeclared component-pair list.
- `gaussian_nll_per_dimension`: the exact Gaussian logarithmic score per
  registered query dimension. Covariances must be symmetric positive definite
  and satisfy the registered condition-number limit; no jitter, clipping, or
  pseudoinverse repair is applied.
- `weighted_interval_score`: the component-mean weighted interval score over a
  common increasing set of central interval coverages.

Energy and variogram scores consume predictive samples. The Gaussian score
consumes a mean and complete covariance in the registered query space. Weighted
interval score consumes a median and central intervals. A scoring bundle may
request any canonical subset, but every matched arm must supply the inputs for
all requested scores.

## Matched input contract

The input contract is
`bayesian-phystwin-probabilistic-score-input-v1`. Every statistical unit must
contain exactly the same sorted method set and the same interval coverages. The
fallback arm must be present and marked accepted. Rejected candidate arms retain
their raw score for diagnosis, but their deployed score is replaced by the
registered fallback arm's score.

Optional `comparison_pairs` bind named candidate-minus-reference contrasts.
The report records raw and deployed differences, equal-group means, and
unit-level win/tie/loss counts. Lower scores are better, so a negative difference
favors the candidate.

The complete input is content-addressed by the CLI output. Use complete physical
objects or independent acquisition sessions as `group_id`; do not use frames,
points, views, or tracks as independent groups.

## CLI

Run the scorer through the grouped diagnostic surface:

```bash
bpt diagnostic run score-probabilistic-predictions \
  examples/bayesian_value_decomposition_score_input.json \
  outputs/probabilistic-score-report.json \
  --evidence-json outputs/probabilistic-decisive-evidence.json
```

The report contains raw and deployed proper scores, exact-fallback status,
interval diagnostics, equal-group summaries, pairwise attribution, and a
content identity. The optional evidence output can be passed directly to:

```bash
bpt evidence summarize \
  outputs/probabilistic-decisive-evidence.json \
  outputs/probabilistic-decisive-summary.json \
  --reference-method last_residual
```

The reader rejects duplicate JSON keys, non-finite values, symlinks, changing
files, and inputs above the byte budget. Outputs are published atomically and do
not overwrite by default.

## Negative Gaussian log scores

A Gaussian density can exceed one in physical units, so a valid logarithmic
score can be negative. The decisive-evidence v1 schema represents losses as
nonnegative values. During evidence conversion, the adapter adds one recorded
common constant to every method for each unit and score. This preserves method
ordering, all paired differences, and exact-fallback equality. The original
unshifted proper scores remain authoritative in the score report.

## Claim boundary

A low score is useful evidence only under a separately frozen cohort, query,
method, information boundary, and decision rule. The score report itself always
sets `claim_authorized=false`.
