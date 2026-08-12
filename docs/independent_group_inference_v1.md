# Independent-group inference v1

## Purpose

`bayesian_phystwin.independent_group_inference_v1` records the statistical
analysis for a frozen paired comparison over genuinely independent physical
objects or acquisition sessions. It is intended for prospective validation in
which frames, horizons, tracks, points, camera views, and tactile taxels are
repeated observations inside one group rather than additional independent
samples.

The input is one matrix

```text
group_effects[group, estimand]
```

whose entries are already frozen candidate-minus-comparator effects. Negative
values favor the candidate. Each row receives equal weight. Any within-group
aggregation, such as an equal-horizon mean Gaussian-NLL difference, must be
fixed before target outcomes are opened and is bound verbatim in the artifact.

## Statistical operations

The contract performs two complementary operations.

### Exact paired sign-flip randomization

For `G` independent groups, all `2**G` joint sign patterns are enumerated. The
per-estimand statistic is the mean effect divided by its root-mean-square group
effect. The denominator is invariant to sign flips, avoids a hidden variance
floor, and makes familywise comparisons scale invariant.

The result records:

- exact one-sided unadjusted p-values;
- single-step familywise p-values based on the minimum statistic over the
  preregistered estimand family; and
- one global family p-value.

This test requires joint sign symmetry of the complete-group effects under the
null. It is not justified merely because there are many frames or points inside
each object. Version 1 deliberately fails when more than 20 independent groups
would require approximate rather than exact enumeration.

### Deterministic paired group bootstrap

One `numpy.PCG64` index stream resamples complete group rows with replacement.
The exact same sampled group indices are used for every estimand. The artifact
records the seed, replicate count, fixed chunking policy, and SHA-256 digests of
both the generated index stream and the resulting bootstrap means.

The bootstrap reports:

- pointwise percentile intervals;
- two-sided simultaneous intervals from the maximum absolute standardized mean
  deviation; and
- one-sided simultaneous upper bounds for the negative-is-better superiority
  direction.

The default is 100,000 replicates, matching the currently registered fresh
object/session analysis. Workload limits cap the exact family, bootstrap draws,
and stored bootstrap-result values before allocation.

## Content-addressed record

`IndependentGroupInferenceV1` canonicalizes group and estimand order by exact
identity, defensively owns all arrays, freezes finite JSON metadata, replays every
statistical result during loading, and assigns a SHA-256 `artifact_id`. The JSON
record includes the complete group-effect matrix and all summary results while
binding the unstored resampling draws through their digests.

Changing a group, estimand, effect, seed, family definition, aggregation rule,
interval level, resampling result, or metadata changes the artifact identity.
Unknown fields, duplicate JSON keys, non-finite constants, and altered summary
values are rejected.

## Example

```python
import numpy as np

from bayesian_phystwin.independent_group_inference_v1 import (
    analyze_independent_group_inference_v1,
    save_independent_group_inference_v1,
)

result = analyze_independent_group_inference_v1(
    protocol_id="registered-protocol-content-id",
    family_id="candidate-vs-two-comparators-v1",
    statistical_unit="complete physical object-session",
    within_group_aggregation="equal-object/equal-horizon Gaussian NLL mean",
    group_ids=tuple(f"object-{index:02d}" for index in range(12)),
    estimand_ids=(
        "candidate-minus-last-residual",
        "candidate-minus-physical-fallback",
    ),
    group_effects=np.asarray(group_level_nll_differences),
    bootstrap_replicates=100_000,
    bootstrap_seed=20260812,
    metadata={
        "point_mean_identity": "exact-last-residual",
        "target_side_retuning": False,
    },
)

save_independent_group_inference_v1(
    result,
    "outputs/independent-group-inference-v1.json",
)
```

A simultaneous superiority upper bound below zero is evidence in the declared
direction only when the complete preregistered information order, cohort,
comparator, candidate, guard, fallback, and estimand family are independently
satisfied. This artifact does not itself authorize target access, candidate
selection, claim promotion, deployment, or a state-of-the-art statement.

## Scientific boundary

The module is analysis and provenance infrastructure. It cannot repair an
opened cohort, create independent groups from repeated measurements, justify a
post-outcome exclusion, or convert a retrospective result into confirmation.
A negative result is a complete result under the frozen design and must not be
retuned on the same target groups.
