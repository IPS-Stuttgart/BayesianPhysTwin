# Tracking Cloth query-quotient real-data validation

This experiment instantiates the query-quotient theory on the complete public
Tracking Cloth Deformation dataset, Zenodo record `14644526`. The retained cache
contains the verified publisher archive and all 120 extracted recordings.

## Scientific question

Thirty-two shaking recordings update one physical-hypothesis belief for each of
eight material-size specimens. That belief is transported to four twisting
conditions per specimen. For every target condition, the frozen nine-member
stiffness/damping bank is partitioned into low, middle, or high nonrigid
deformation regimes using only model predictions generated from the target
initialization prefix and the prescribed driven-corner trajectory.

The registered categorical query is therefore available before any future free
marker is scored. Its class posterior is lifted to complete hypothesis beliefs in
five different ways:

- the full source posterior;
- the Jeffrey or forward-I-projection lift;
- a uniform-within-class lift;
- prior-MAP concentration within each class;
- prior-antimap concentration within each class.

All five beliefs must have identical quotient masses and therefore identical
categorical log and Brier scores. Their continuous deformation expectations,
latent parameter decisions, mean trajectories, and trajectory energy scores may
differ. The Jeffrey lift is distinguished by zero unsupported within-class
specificity, not by an assumption that it minimizes held-out trajectory error.

The registered physical prior is nonuniform over the existing 3 x 3 parameter
bank: `[0.2, 0.6, 0.2]` independently for stiffness and damping. This makes the
Jeffrey lift different from a uniform-within-class allocation. A cyclic
wrong-specimen posterior and a reversed hypothesis-identity control test whether
source-to-target gains survive broken physical relations.

## Information order

1. Audit the complete dataset and verify extracted bytes against the retained
   publisher ZIP.
2. Read the 32 shaking recordings and compute one normalized loss per recording
   and hypothesis.
3. Form eight prior-aware source posteriors.
4. Read only each twisting prefix and future prescribed corner coordinates.
5. Generate all target hypothesis trajectories, query partitions, quotient
   posteriors, comparison lifts, and private prediction arrays.
6. Upload the complete prediction seal before opening future free-marker target
   values.
7. Score all 32 targets, then aggregate four conditions within each of eight
   material-size specimens.

Raw recordings and private target trajectory arrays are not uploaded. A failure
before prediction-seal upload prevents target scoring.

## Endpoints

The primary query endpoints are categorical log score and Brier score for the
observed low/middle/high tail-deformation regime. Supporting endpoints are class
accuracy, a finite-distribution trajectory energy score, mean-trajectory RMSE,
continuous-query absolute error, quotient information, unsupported specificity,
and exact ambiguity-envelope widths. Exploratory paired intervals resample the
eight specimens; frames and marker coordinates are not treated as independent
replicates.

## Claim boundary

This is retrospective public-real-data validation. Prior public target exposure
is not assumed absent, so the run is not fresh confirmation. Future measured
corner trajectories are prescribed inputs; this is not command-conditioned
forecasting. The finite bank is a transparent spring-mesh baseline, not unique
material identification or high-fidelity cloth mechanics. The study does not
authorize a paper claim automatically and does not establish unseen-object
transport, calibrated joint trajectory covariance, causal intervention effects,
or safety.

## Execution

The file-triggered workflow runs on
`[self-hosted, Linux, X64, gpuserver4090]` and reads:

```text
/home/github-runner/.cache/datasets/tracking-cloth-deformation-v1-zenodo-14644526
```

The trigger is `run_requests/tracking-cloth-query-quotient-real-v1.json`.
