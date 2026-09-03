# Tracking Cloth matched-coverage uncertainty ablation v1

This retrospective study leaves the existing Tracking Cloth candidate and exact
persistence fallback unchanged. It asks whether an uncertainty-bearing ranking
adds selective value when every method accepts exactly 32 of 320 query cases in
each held-out material (10% coverage).

The primary comparison is a leave-one-material-out ridge selector using
`motion × query × horizon`, size, speed, grasp, and ensemble spread against the
same source-fitted context model without spread. Target-material outcomes never
enter non-oracle fitting or ranking. Candidate, MAP, last-residual, and nominal
arms are also evaluated under each identical selection mask.

Complete recordings are the bootstrap units; query and horizon cases remain
nested. This is a retrospective ablation on outcomes opened before the study,
not fresh confirmation, calibration, control safety, or state-of-the-art
evidence.

Run after regenerating the compact selective-twin output:

```bash
python -m experiments.tracking_cloth_matched_coverage_v1.run \
  --query-csv base/query_cases.csv \
  --policy-csv base/policy_cases.csv \
  --base-result-json base/result.json \
  --output-dir output
```
