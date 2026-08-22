# Fixed-mean Gaussian NLL diagnostic v1

## Purpose

This source-only diagnostic explains **why** one covariance changes the Gaussian
negative log predictive density when both arms use the exact same predictive
mean. It is intended for already-open development or source evidence, including
retrospective analysis of covariance-only BayesianPhysTwin arms.

It does not alter the frozen Deform360 independent-validation protocol, select a
new covariance, authorize target access, or create a scientific claim.

## Decomposition

For a registered query observation `y`, shared predictive mean `mu`, covariance
`Sigma`, and query dimension `d`, the per-dimension Gaussian negative log score
is

```text
NLL / d
  = 0.5 log(2 pi)
  + 0.5 log det(Sigma) / d
  + 0.5 (y - mu)^T Sigma^-1 (y - mu) / d.
```

The analyzer calls the second term the **sharpness term** and the third term the
**standardized-error term**. Because the input has only one mean field, both
covariance arms necessarily use the same mean and the normalization term
cancels exactly:

```text
candidate NLL - reference NLL
  = sharpness difference + standardized-error difference.
```

The report verifies this identity for every record and again after aggregation.

## Interpretation

All differences use `candidate covariance minus reference covariance` semantics.
Lower NLL is better.

| Quantity | Interpretation |
| --- | --- |
| NLL difference below zero | The candidate covariance obtains the better Gaussian score. |
| Sharpness difference above zero | The candidate pays a larger predictive-volume penalty. |
| Standardized-error difference below zero | The candidate covariance better accommodates the fixed-mean residuals. |
| Width ratio above one | Candidate marginal intervals are wider on average. |
| Coverage nearer the nominal level | Descriptive support for marginal calibration, not proof of calibration. |

A favorable standardized-error term can arise from useful covariance direction,
appropriate scale, or indiscriminate inflation. The simultaneous sharpness,
coverage, and interval-width outputs make those explanations distinguishable.
They do not identify a physical latent state or establish downstream utility.

## Statistical weighting

The input records must form a complete rectangular `group x horizon` roster.
The analyzer applies three explicit stages:

1. equal weight to records within each group-horizon cell;
2. equal weight to horizons within each complete group; and
3. equal weight to complete groups in the overall summary.

Frames, coordinates, vertices, cameras, and repeated rows therefore cannot
silently increase the number of independent groups. The report records this
weighting policy and a numerical tolerance used only to classify effectively
zero group-level NLL differences.

## Input contract

The root object uses contract
`bayesian-phystwin.fixed-mean-gaussian-nll-diagnostic-input-v1` and requires:

- source/protocol/query/observation-model identities;
- distinct reference and candidate arm identities;
- `retrospective-source-only-non-claim-bearing` analysis status;
- `claim_authorized: false`;
- the independent statistical-unit label;
- a canonical horizon order;
- nominal marginal coverage in `(0.5, 1.0)`;
- a maximum covariance condition number; and
- sorted, complete records.

Each record contains one mean and observation plus the reference and candidate
covariance matrices. Covariances must be finite, symmetric positive definite,
and within the registered condition-number limit.

A minimal synthetic input is:

```json
{
  "analysis_id": "example-fixed-mean-source-diagnostic-v1",
  "analysis_status": "retrospective-source-only-non-claim-bearing",
  "candidate_arm_id": "same_mean_with_candidate_covariance",
  "claim_authorized": false,
  "contract": "bayesian-phystwin.fixed-mean-gaussian-nll-diagnostic-input-v1",
  "horizon_order": ["early"],
  "maximum_condition_number": 100000000.0,
  "nominal_coverage": 0.9,
  "observation_model_id": "gaussian-observation-5mm-v1",
  "protocol_id": "already-open-source-analysis-v1",
  "query_id": "registered-query-v1",
  "records": [
    {
      "candidate_covariance": [[4.0, 0.0], [0.0, 1.0]],
      "group_id": "object-session-a",
      "horizon": "early",
      "mean": [0.0, 0.0],
      "observation": [2.0, 0.0],
      "reference_covariance": [[1.0, 0.0], [0.0, 1.0]],
      "unit_id": "object-session-a/early/0"
    }
  ],
  "reference_arm_id": "same_mean_with_reference_covariance",
  "schema_version": 1,
  "scientific_boundary": "already-open source evidence; no claim",
  "source_artifact_id": "sealed-source-score-table-v1",
  "statistical_unit": "physical-object-session"
}
```

Records must be ordered by `group_id`, the declared `horizon_order`, and
`unit_id`.

## Command

```bash
python scripts/science/analyze_fixed_mean_gaussian_nll_v1.py \
  source_input.json \
  --output fixed_mean_nll_report.json
```

The command refuses to overwrite an existing report unless `--force` is
provided. Publication is atomic. The output contains the input content identity,
a content-addressed `report_id`, per-record terms, group-by-horizon summaries,
equal-group summaries, coverage, interval width, and explicit claim boundaries.

## Scientific boundary

This diagnostic is explanatory infrastructure. It may support a statement such
as “the fixed-mean NLL gain was dominated by reduced standardized error despite
a stated sharpness and width cost” on already-open source evidence. It cannot by
itself support covariance selection, independent confirmation, physical-state
identification, provider competence, unseen-object transfer, Causal4D benefit,
deployment safety, or state of the art.
