# PhysicalQueryV1

`PhysicalQueryV1` is the BayesianPhysTwin-owned, target-blind definition of the
physical quantity and decision policy used by a fresh Prob4D provider-promotion
study. It closes a recurring ambiguity: a valid Prob4D observation artifact is
not itself permission to update a physical twin, and good provider-space scores
do not by themselves establish downstream physical benefit.

The intended ownership chain is:

```text
Prob4D observation and structured covariance
                    |
                    v
BayesianPhysTwin PhysicalQueryV1 and accept/fallback decision
                    |
                    v
Causal4D may consume only the accepted belief
```

## Contract contents

The content-addressed query binds all quantities that could otherwise drift after
target outcomes are opened:

- ordered query components, dimension, physical unit, coordinate frame, and
  strictly increasing horizon values;
- the exact Jacobian provider, baseline physical belief, and fallback-byte
  identities;
- both the marginal-gauge and complete explicit-joint-gauge covariance
  treatments, together with the source-selected principal treatment;
- the primary proper score and frozen practical-equivalence, harmful-update,
  coverage, interval-width, worst-group-regret, and shared-covariance margins;
- the query-space shared-covariance diagnostic and any computational selection
  rule;
- the complete physical object/session definition, resampling method, seed,
  confidence level, and declared strata;
- exact package artifacts, repository revisions, provider manifest, and source
  evidence-decision identities; and
- a fixed information boundary assigning update admission to BayesianPhysTwin
  and prohibiting target-informed method selection or retuning.

Every bound repository must be clean. The repository roster must contain exactly
one `IPS-Stuttgart/BayesianPhysTwin` primary and exactly one
`IPS-Stuttgart/Prob4D` observation provider. Package bindings must include at
least `bayesian-phystwin` and `prob4d`.

A valid `query_id` is a prerequisite for a target execution, not an execution or
claim authorization. The separately registered prediction, target-access,
scoring, and evidence-decision records remain mandatory.

## Example

```python
from bayesian_phystwin.v1 import (
    PhysicalQueryBootstrapV1,
    PhysicalQueryDecisionMarginsV1,
    PhysicalQueryV1,
    RepositoryState,
    write_physical_query,
)

query = PhysicalQueryV1(
    query_name="fresh-provider-endpoint-displacement",
    dimension=6,
    component_order=(
        "early-x",
        "early-y",
        "early-z",
        "late-x",
        "late-y",
        "late-z",
    ),
    physical_unit="m",
    coordinate_frame="registered-world-frame",
    horizon_values=(0.08, 0.20),
    horizon_unit="s",
    jacobian_provider_id="<sha256>",
    baseline_physical_belief_id="<sha256>",
    exact_fallback_id="<sha256>",
    covariance_treatments=(
        "marginal-gauge",
        "complete-explicit-joint-gauge",
    ),
    principal_covariance_treatment="complete-explicit-joint-gauge",
    primary_proper_score="gaussian_nll_per_dimension",
    decision_margins=PhysicalQueryDecisionMarginsV1(
        practical_equivalence_score=0.001,
        maximum_harmful_score_increase=0.005,
        minimum_accepted_coverage=0.85,
        maximum_mean_width=0.050,
        maximum_worst_group_score_regret=0.010,
        minimum_shared_covariance_relevance=0.05,
        width_unit="m",
    ),
    shared_covariance_diagnostic=(
        "query trace fraction attributable to shared gauge covariance"
    ),
    computational_selection_rule=(
        "use the complete joint treatment unless its source-only relevance "
        "falls below the frozen threshold"
    ),
    bootstrap=PhysicalQueryBootstrapV1(
        independent_group_definition="complete physical object/session",
        method="paired-stratified-group-bootstrap",
        resamples=10_000,
        seed=1731,
        confidence_level=0.95,
        stratification_keys=("object-stratum", "action-family"),
    ),
    package_artifact_ids={
        "bayesian-phystwin": "<sha256>",
        "prob4d": "<sha256>",
    },
    provider_manifest_id="<sha256>",
    evidence_decision_ids={"source-provider-gate": "<sha256>"},
    repositories=(
        RepositoryState(
            repository="IPS-Stuttgart/BayesianPhysTwin",
            revision="<40-character-git-sha>",
            dirty=False,
            role="primary",
        ),
        RepositoryState(
            repository="IPS-Stuttgart/Prob4D",
            revision="<40-character-git-sha>",
            dirty=False,
            role="observation",
        ),
    ),
)
write_physical_query(query, "physical-query.json")
```

The writer refuses replacement by default and publishes canonical JSON
atomically. The loader rejects duplicate JSON keys, unknown or missing fields,
non-finite values, changed ownership or target boundaries, dirty repositories,
missing required covariance treatments, and a mismatched content identifier.

## Scientific boundary

A valid query establishes only that the intended physical quantity, comparison,
statistical unit, decision margins, and software/evidence identities were fixed
before target outcomes. It does not establish provider competence, physical
benefit, calibrated deployment uncertainty, Causal4D intervention benefit,
deployment safety, or state of the art. Those require a separately registered,
sealed, independently scored target execution with exact fallback accounting.
