# Evidence decision v1

`EvidenceDecisionV1` is a compact, content-addressed `decision.json` contract for
one already-evaluated scientific gate. It does not replace numerical results,
tables, figures, the run manifest, or the claim bundle. It gives reviewers and
automation a small fail-closed record that says what was decided and binds that
record to the exact evidence.

The contract records:

- a claim and protocol identifier;
- `pass`, `fail`, `degraded`, or `inconclusive` status;
- the run classification that governs claim promotion;
- whether the result is allowed to authorize the scoped claim;
- evidence level 1, 2, or 3;
- one primary scalar metric, comparison, and frozen rule;
- the run-manifest ID, evidence fingerprint, and evidence-summary SHA-256;
- exact participating repository revisions and dirty states; and
- limitations plus finite JSON metadata.

Claim authorization fails closed: only a passing decision may authorize a claim,
and an authorized decision cannot bind a dirty repository. Degraded and
inconclusive results must state at least one limitation. Loading rejects unknown
or missing fields, duplicate JSON keys, non-finite values, invalid revisions,
and content-address drift.

## Preferred construction from a run manifest

The builder derives all provenance identities from a finalized `RunManifestV2`.
It requires the claim and protocol to be declared by that manifest and verifies
that the evidence summary is an exact manifest output. An authorized claim also
requires a clean confirmatory run.

```python
from bayesian_phystwin.v1 import (
    DecisionMetricV1,
    build_evidence_decision,
    load_run_manifest,
    write_evidence_decision,
)

manifest = load_run_manifest("result/manifest.json")
decision = build_evidence_decision(
    manifest=manifest,
    evidence_summary_path="result/summary.json",
    claim_id="bpt.physical.guard",
    status="pass",
    claim_authorized=True,
    evidence_level=3,
    metric=DecisionMetricV1(
        name="future_track_error_mm",
        comparison="action_discrepancy_vs_nominal",
        rule="relative_improvement_gt_0",
        observed_value=13.52,
        threshold_value=0.0,
        unit="percent",
    ),
    limitations=("independent-object confirmation only",),
    metadata={"profile": "publication"},
)
write_evidence_decision("result/decision.json", decision)
```

The writer uses canonical JSON and refuses to replace an existing decision unless
`overwrite=True` is explicit. The decision can be added to a `ClaimBundleV1` as
a `supporting` artifact, avoiding a circular dependency between the decision ID
and bundle ID.

## Wire schema

The installed package carries the normative data-only JSON Schema at
`contract_data/evidence_decision_v1/evidence-decision-v1.schema.json`. The
schema fixes the closed wire shape. Python validation additionally enforces
semantic constraints that JSON Schema cannot conveniently express, including
unique repository identities and content-address verification.

## Versioned integration API

New cross-repository consumers should import portable contracts from
`bayesian_phystwin.v1`. The legacy package-root API remains unchanged. The v1
namespace intentionally exports only observation, provenance, run-manifest,
claim-bundle, and decision contracts plus their load/write helpers.
