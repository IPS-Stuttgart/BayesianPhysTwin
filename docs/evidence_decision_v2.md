# Evidence decision v2

`EvidenceDecisionV2` is the claim-bearing decision envelope for runs whose
technical execution, scientific outcome, and authorization outcome must remain
separate. It preserves the content-addressed manifest, evidence-summary, and
multi-repository bindings of v1. Existing v1 JSON is unchanged and remains
loadable.

## Why v2 exists

The v1 `status` field combines several materially different terminal states. In
particular, a valid scientific negative, a runner failure, and an invalid
protocol must not all appear as a generic failed workflow. They have different
reproducibility, retry, reporting, and publication consequences.

V2 records three explicit axes:

| Axis | Values | Meaning |
| --- | --- | --- |
| `execution_status` | `completed`, `infrastructure_failure`, `protocol_invalid` | Whether the registered execution was technically and procedurally valid |
| `scientific_decision` | `pass`, `negative`, `not_evaluated` | Outcome of the registered scientific rule, without inferring authorization |
| `authorization` | `advance`, `stop` | Whether the bound evidence permits the claim or protocol to advance |

The metric is present only when the scientific rule was evaluated. A completed
infrastructure or diagnostic run may therefore be `completed`,
`not_evaluated`, and `stop` with a required limitation and `metric: null`.

## Fail-closed invariants

The Python implementation and packaged JSON Schema enforce the same core rules:

- non-completed executions are `not_evaluated`, have no metric, stop, and record
  at least one limitation;
- scientific negatives and non-evaluated runs always stop;
- evaluated `pass` and `negative` decisions require a finite registered metric;
- a passing result stopped by policy records the reason as a limitation;
- advancement requires a completed passing confirmatory run and clean bindings
  for every repository;
- the decision ID covers all outcome, metric, provenance, limitation, and
  metadata fields.

This makes a valid negative a successful technical artifact:

```json
{
  "execution_status": "completed",
  "scientific_decision": "negative",
  "authorization": "stop"
}
```

An execution failure is represented without claiming a scientific result:

```json
{
  "execution_status": "infrastructure_failure",
  "scientific_decision": "not_evaluated",
  "authorization": "stop",
  "metric": null
}
```

## Python API

Build and publish v2 explicitly:

```python
from bayesian_phystwin.evidence_decision_v2 import (
    DecisionMetricV2,
    build_evidence_decision_v2,
    write_evidence_decision_v2,
)

metric = DecisionMetricV2(
    name="mean_gaussian_nll_delta",
    comparison="candidate_vs_exact_fallback",
    rule="delta_lt_0",
    observed_value=-0.18,
    threshold_value=0.0,
    unit="nat_per_group",
)

decision = build_evidence_decision_v2(
    manifest=manifest,
    evidence_summary_path="evidence-summary.json",
    claim_id="bpt.independent_object_predictive_value",
    execution_status="completed",
    scientific_decision="pass",
    authorization="advance",
    evidence_level=3,
    metric=metric,
)
write_evidence_decision_v2("evidence-decision.json", decision)
```

Consumers that accept both versions should use the version-dispatching loader:

```python
from bayesian_phystwin.evidence_decision import load_evidence_decision

decision = load_evidence_decision("evidence-decision.json")
```

The loader dispatches only from the exact schema name and integer version. It
does not guess a version from field presence.

## Migration from v1

V1 artifacts are historical records and must not be rewritten. New decisions
should use v2. When interpreting an old artifact, use the following only as a
human migration guide; do not silently mutate its content identity:

| V1 state | Typical v2 interpretation |
| --- | --- |
| `pass`, authorized | `completed` / `pass` / `advance` |
| `pass`, not authorized | `completed` / `pass` / `stop`, with the policy limitation |
| `fail` after a valid endpoint evaluation | `completed` / `negative` / `stop` |
| `degraded` or `inconclusive` | classify from the retained run record; do not infer whether execution or science failed |

The last row is intentionally not automatic. V1 cannot always distinguish an
invalid execution from a valid but non-decisive scientific result, and v2 must
not fabricate that distinction.

## Cross-repository use

Prob4D producers and Causal4D consumers can bind their exact revisions through
the existing repository roles. A single decision can therefore distinguish:

1. whether the Prob4D evidence production completed validly;
2. whether the registered Bayesian-PhysTwin rule passed; and
3. whether Causal4D or the paper pipeline may advance the claim.

Downstream automation should branch on the explicit fields rather than on the
GitHub Actions conclusion. A valid scientific negative should normally leave
the workflow technically successful while publishing `authorization: stop`.
