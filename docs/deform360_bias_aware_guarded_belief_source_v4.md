# Deform360 Bias-Aware Guarded Belief: Source v4

## Status

This is an **already-open source-development result**, not a prospective test.
The 27 episodes and five objects were available before v4 was selected. The
result may lock a candidate for a genuinely fresh evaluation, but it cannot
establish calibration, safety, state of the art, or non-regression on new
objects.

The method addresses the failure exposed by the prospective camera-only
virtual-sensing study. Under a coherent observation model

```text
y = d + b + e,
```

camera evidence alone cannot distinguish real displacement `d` from shared
bias `b`. V4 therefore admits an update only when the observed prefix has
target-free support from the simulated physical response, removes state modes
confounded with the declared observation-bias basis, and preserves the exact
selected baseline everywhere else.

## Frozen Candidate

The v4 candidate uses:

- update frames 19, 38, and 57;
- at least nine available centres and three motion centres;
- at least 0.5 mm physical response and 0.5 mm observed motion;
- robust physical-response agreement of at least 0.40;
- a rank-4 causal physical-response basis;
- residual-independent triangulation reliability and metric cycle covariance;
- one Student-t innovation update;
- explicit shared spatial and global camera-bias terms;
- at least 10% identifiable state support beyond the bias basis;
- a frozen source-group regret bound with a required 0.005 mm improvement;
- bit-exact selected-raw-baseline fallback.

Candidate construction accepts no target and reads no future observation. The
Deform360 adapter decodes the inferred response-constrained state coefficients
as a persistent low-rank correction to the selected raw trajectory. It does
**not** claim that Warp was restarted from a corrected simulator state in this
experiment.

## Development Trail

All rows below are post-open development and must be treated jointly. V1 and V2
report the unguarded candidate because the direct source-group guard had not
yet been introduced; V3 and V4 report the group-bound arm.

| Version | Minimum physical agreement | Eligible intervals | Identity change | Chamfer change |
| --- | ---: | ---: | ---: | ---: |
| v1 | none | 51 | +0.754% | -0.336% |
| v2 | 0.10 | 19 | -1.528% | -1.415% |
| v3 | 0.35 | 11 | -1.414% | -1.330% |
| v4 | 0.40 | 10 | -1.414% | -1.330% |

The v1 regression showed that bias modeling alone was insufficient. V2 added
the causal physical-agreement shrinkage suggested by the prospective failure.
The v3-to-v4 change deliberately gives up one tiny accepted update while
retaining essentially all aggregate benefit and creating a nonzero source
regret margin. The 0.40 threshold was selected after inspecting these open
source outcomes, so only a new cohort can test it.

## V4 Result

Primary values are object-balanced means against the already selected raw
AllTracker/physical-persistence backbone.

| Metric | Selected baseline | V4 guarded | Difference | Relative | Object-cluster 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hidden identity RMSE | 8.807 mm | 8.683 mm | -0.125 mm | -1.414% | [-0.271, -0.024] mm |
| Hidden symmetric Chamfer | 7.888 mm | 7.783 mm | -0.105 mm | -1.330% | [-0.208, -0.012] mm |

At the episode level, both metrics have 7 wins, 20 exact ties, and no losses.
The object-held-out direct group bound accepts 10 of 81 update intervals and
falls back exactly on the remaining 71. No accepted interval is harmful on
either primary metric. The feature-conditional ridge certificate accepts no
updates; with only five source objects, its uncertainty is too large. V4
therefore freezes the simpler eligibility rule plus direct source-group bound.

## Calibration Boundary

Leave-one-object-out fits have only three eligible source groups. Their exact
finite-sample rank is 3 of 4, or 75%, despite the requested 90% level. The
full-source lock has four eligible groups and a rank of 4 of 5, or 80%. Its
worst source-group regret is `-0.0088711363 mm`, which clears the frozen
`0.005 mm` improvement requirement, but this is not a 90% guarantee.

The bound is conditional on both:

1. the frozen target-free eligibility rule; and
2. exchangeability with the four eligible source objects.

It does not protect against arbitrary coherent camera bias or unrestricted
domain shift. The exact fallback protects implementation behavior when the
candidate is rejected; it does not turn a source-fitted acceptance decision
into a universal theorem.

## Decision

A genuinely fresh, preregistered **accuracy and non-regression** evaluation is
justified. A calibrated 90% safety claim is not. The fresh protocol must:

1. exclude every object in the open source and selective-virtual-sensing
   cohorts, plus every reserved or confirmatory object;
2. select objects and episodes from metadata only;
3. consume the committed `prospective_lock.json` without changing any method
   choice;
4. seal baseline and v4 prediction artifacts before opening targets;
5. use object-clustered identity RMSE and Chamfer as co-primary outcomes;
6. report acceptance, exact fallback, harmful updates, and all quality failures;
7. make no 90% calibration or state-of-the-art claim from the source lock.

## Artifacts

- Source summary:
  `results/sota/deform360_bias_aware_guarded_belief_v4/summary.json`
  (SHA256 `dbad5fd3b4d572d515d38b9bb31df84a2f036c223aaed3aa0810c25fbec3e015`)
- Prospective candidate lock:
  `results/sota/deform360_bias_aware_guarded_belief_v4/prospective_lock.json`
  (SHA256 `5f5672d35aa41e276f1dd5ace54b6694b0139ff2a562e3c3a24558fa555c9dd6`)
- Interim v3 summary:
  `results/sota/deform360_bias_aware_guarded_belief_v3/summary.json`
  (SHA256 `1658064199a89e63fab56c8672e447b1eecc7c58b0df934624f1bbd0cee52054`)

The complete per-case artifacts remain on `gpuserver6000` at
`/mnt/corsair/florianpfaff/bpt-bias-aware-open27-v4-locked-development`.
