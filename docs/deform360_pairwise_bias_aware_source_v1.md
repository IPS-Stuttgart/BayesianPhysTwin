# Pairwise Bias-Aware Belief: Source V1

## Status

Completed source-development negative result. The combined pairwise
bias-aware candidate failed its frozen advancement gates. No fresh-object
evaluation is justified.

This route may use only the 27 already-open Deform360 development episodes.
It must not inspect held-v8, the failed selective-virtual-sensing target cohort,
or any fresh object. A source result can justify a new prospective protocol; it
cannot establish state of the art, calibration, or non-regression.

## Question

The strongest open-source camera result uses a pairwise-consensus gate followed
by a persistent RBF correction. It improves hidden identity RMSE and hidden
Chamfer substantially, but a rigid-invariant clique cannot detect coherent
camera bias. The separate bias-aware physical-response update rejects that
ambiguity, but its open-source gain is much smaller.

The v1 candidate asks whether those mechanisms compose:

```text
causal physical response
        + pairwise material-identity consistency
        + nuisance-marginalized observation selection
        + bias-aware state update
        -> guarded state correction or exact baseline
```

## Frozen Candidate

At update frames 19, 38, and 57:

1. Start from the already selected physical/persistence backbone.
2. Admit only finite observations with positive residual-independent prior
   reliability.
3. Find the frozen pairwise-consensus clique using the existing 30 mm plus 10%
   strain envelope, minimum nine inliers, and 70% support.
4. Require at least three causal motion centers, 0.5 mm physical response,
   0.5 mm observed motion, and physical-response agreement of at least 0.40.
5. Build the rank-4 physical-response basis from prefix-only simulated motion.
6. Remove physical modes that are less than 10% identifiable beyond the shared
   spatial and global-bias basis.
7. Select at most 12 clique members by nuisance-marginalized information gain.
   The total reliability mass is capped at eight effective observations.
8. Run the existing Student-t bias-aware update once and propagate only its
   reachable state component. Coherent shared and global bias is not added to
   the physical state.
9. Multiply the decoded state correction by the causal physical-agreement gain.
10. On every rejection or numerical failure, return the selected backbone
    byte-for-byte.

Candidate geometry determines association consistency. Prior reliability uses
triangulation redundancy, reprojection quality, and covariance, never the
innovation against the twin. The state innovation enters once through the
bias-aware robust likelihood.

## Synthetic Controls

The implementation must continue to pass:

- three large identity swaps are excluded by the pairwise clique;
- a 10 mm coherent observation translation is assigned to nuisance bias while
  the local physical mode is recovered;
- a common translation without matching physical response gives exact fallback;
- changing a global state innovation does not alter prior selection reliability;
- center-array permutation does not change selected material identities;
- duplicated/correlated evidence cannot exceed the fixed effective information
  mass;
- insufficient pairwise support gives byte-exact fallback.

These are implementation and mechanism controls, not empirical evidence.

## Frozen Source Arms

The source run must report the same hidden identities and future frames for:

| Arm | Purpose |
| --- | --- |
| Selected raw backbone | Exact no-update reference |
| Pairwise-consensus RBF | Strong open-27 camera baseline |
| Bias-aware v4 | Physical-support and nuisance-model control |
| Pairwise bias-aware v1 | New candidate |

Report object-balanced hidden identity RMSE, hidden symmetric Chamfer, late
errors, update coverage, exact-fallback count, selected-center count,
state/bias posterior correlation, and object-cluster bootstrap intervals.

## Advancement Gates

A new fresh-object accuracy protocol is justified only if all conditions hold:

1. both co-primary object-balanced metrics improve by at least 1% over the
   pairwise-consensus RBF arm;
2. both object-clustered 95% difference intervals have upper endpoints below
   zero;
3. at least four of five object means improve on both metrics;
4. no object regresses by more than 2% on either metric;
5. every rejected interval is byte-identical to its selected backbone;
6. all accepted updates satisfy the frozen pairwise, physical-response,
   identifiability, and information-support gates.

Failure closes this composition on the opened source cohort. Hyperparameters
must not be changed after source outcomes are computed.

## Information Boundary

- no future observation may construct a candidate;
- no target trajectory is accepted by the prediction function;
- open source targets are used only by the separate evaluator;
- the existing held-v8 and fresh-object exclusion boundaries remain unchanged;
- the unavailable held-v8 all-attempt exclusion manifest still blocks any new
  Deform360 object selection.

The implementation entry point is
`predict_pairwise_bias_aware_candidate_arrays` in
`bayesian_phystwin.deform360_pairwise_bias_aware_development`.

The roots must first be staged and validated through the outcome-blind transfer
contract in `docs/deform360_pairwise_bias_aware_transfer_v1.md`. The frozen
comparison then consumes the complete bundle:

```bash
python -m bayesian_phystwin.cli.deform360_pairwise_bias_aware_source \
  --bundle-root /path/to/open27-transfer-v1 \
  --output /new/path/to/source-v1-result
```

## Frozen Source Result

The independently staged bundle contained all 27 cases and 189 bound files.
Its transfer manifest has canonical SHA-256
`02babb4e041fce354a168a76c055a043032dbbf5eb5320e2c8a0e409bb2d83bb`.

| Arm | Identity RMSE | Hidden Chamfer |
| --- | ---: | ---: |
| Selected raw baseline | 8.807 mm | 7.888 mm |
| Single-state pairwise RBF control | 8.304 mm | 7.532 mm |
| Bias-aware v4 | 8.683 mm | 7.783 mm |
| Pairwise bias-aware v1 | 8.707 mm | 7.816 mm |
| Exact dual-backbone pairwise RBF parity audit | **7.441 mm** | **6.795 mm** |

The candidate improves over the selected raw baseline by 1.14% identity RMSE
and 0.92% Chamfer, but it is 4.85% and 3.77% worse than the stronger
pairwise-consensus RBF source arm. Only two of five object means improve
jointly, the clustered intervals do not exclude zero, and the worst object
regresses by 14.93% identity RMSE and 16.54% Chamfer.

All target-free acceptance and exact-fallback checks pass. The failure is
therefore empirical rather than a custody or implementation failure.

## Comparator Parity Audit

The registered source evaluator mislabeled its single-state RBF control as the
earlier strong pairwise-consensus arm. The control receives the already-spliced
selected baseline and keeps one recursive discrepancy state. The original
source arm keeps separate physical and persistence RBF states, then selects
the backbone and matching state causally at each update.

A post-open parity audit replayed the exact dual-backbone implementation from
the transferred `prediction_m` and `persistence_m` arrays. It reproduced the
previous episode means exactly:

- hidden identity RMSE: `7.541936414096939` mm;
- hidden symmetric Chamfer: `6.874515361417309` mm.

Its object-balanced means are 7.441 mm and 6.795 mm. The pairwise bias-aware
candidate is therefore 17.01% worse in identity RMSE and 15.01% worse in
Chamfer than the actual strong source comparator. Correcting the comparator
does not rescue the candidate or authorize a new outcome opening; it makes the
negative decision stronger.

The compact parity record is
`results/sota/deform360_pairwise_bias_aware_source_v1/comparator_parity_audit.json`.

## Decision

This composition is closed on the opened source cohort. Its thresholds must
not be tuned on these five objects, and it does not authorize a fresh-object
evaluation. The simpler pairwise-consensus RBF remains the stronger source
candidate, while common-mode camera bias remains unresolved.

Compact evidence is archived under
`results/sota/deform360_pairwise_bias_aware_source_v1/`.
