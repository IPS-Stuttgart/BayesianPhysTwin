# Source-only provider failure decomposition

`bpt diagnostic run diagnose-provider-failures` turns already-frozen provider
and guarded-update gate outcomes into an equal-case attribution report. The
command explains why an update was rejected without changing thresholds,
rerunning the provider, reading target outcomes, or converting a rejection into
an accepted update.

## Information boundary

This is a diagnostic, not a promotion gate. Inputs must contain only evidence
that was already permitted at the source or calibration stage. The report never
authorizes target access and does not establish provider competence, calibrated
uncertainty, physical-query benefit, deployment safety, Causal4D benefit, or
state of the art.

Unknown evidence remains `null`. It is never interpreted as a passed gate.
Contradictions fail closed: for example, an accepted record cannot declare a
failed gate, and `result_reason: no-identifiable-query-state` cannot be combined
with `query_identifiable: true`.

## Input contract

```json
{
  "schema": "bayesian_phystwin.provider_failure_evidence",
  "schema_version": 1,
  "provider_id": "prob4d-provider-v2-source-lock",
  "records": [
    {
      "case_id": "object-03/session-02",
      "accepted": false,
      "result_reason": "no-identifiable-query-state",
      "signals": {
        "technical_valid": true,
        "provider_support_complete": true,
        "numerically_converged": true,
        "query_identifiable": null,
        "gauge_or_common_mode_consistent": true,
        "covariance_calibrated": true,
        "material_identity_reliable": true,
        "robust_support_sufficient": true,
        "physical_guard_passed": true
      },
      "metrics": {
        "effective_observation_information_mass": 8.4,
        "identifiable_query_state_mode_count": 0
      }
    }
  ],
  "metadata": {
    "split": "source-only"
  }
}
```

Every `case_id` must be unique. Signals are tri-state booleans: `true`, `false`,
or `null`. Metrics and metadata may contain any finite JSON values. Duplicate
JSON keys and non-finite constants are rejected before contract validation.
Frames, points, tracks, views, and taxels should not be duplicated as independent
cases when the registered statistical unit is a physical object or acquisition
session. Record order is part of the portable evidence contract and should be
frozen before publication.

## Failure taxonomy

Primary attribution follows a fixed precedence, while `failed_categories`
retains every observed cause:

1. `technical-failure`
2. `unsupported-provider-geometry`
3. `numerical-non-convergence`
4. `unidentifiable-physical-query`
5. `coherent-gauge-or-common-mode-bias`
6. `provider-covariance-miscalibration`
7. `association-or-material-identity-failure`
8. `outlier-dominated-evidence`
9. `physical-model-or-readout-mismatch`

A rejected record with no explicit or recognized reason-derived failure is
reported as `unresolved-rejection`; it is not guessed into a more favorable
category. Known BayesianPhysTwin reasons such as `no-observation-support`,
`no-identifiable-query-state`, strict-v2 numerical admission failures, singular
posteriors, and `implausible-state-update` provide conservative reason-derived
signals when the corresponding explicit signal is `null`.

## Direct claim-bearing update adapter

Current prospective Prob4D-to-BayesianPhysTwin runs already produce an immutable
`ClaimBearingProb4DUpdateV1` containing provider, calibration, runtime, admission,
and complete inference-result identities. The adapter module converts those
objects directly into the generic diagnostic input contract:

```python
from bayesian_phystwin.provider_failure_decomposition import (
    analyze_provider_failure_evidence,
)
from bayesian_phystwin.provider_failure_evidence_adapters import (
    build_provider_failure_payload_from_claim_bearing_updates,
)

payload = build_provider_failure_payload_from_claim_bearing_updates(
    [
        ("object-03/session-02", update_03_02),
        ("object-04/session-01", update_04_01),
    ],
    source_signals_by_case={
        "object-03/session-02": {
            "covariance_calibrated": False,
        }
    },
    metadata={"split": "source-only"},
)
report = analyze_provider_failure_evidence(payload)
```

The adapter accepts only a strict `PriorAwareGaugeBeliefResultV2` carried by a
validated claim-bearing update. It derives only facts established by that
immutable record:

- technical construction is valid;
- provider support is failed for recognized support reasons, or passed when the
  underlying update was admissible;
- numerical convergence is passed only by a complete strict admission and failed
  by a recognized strict numerical reason;
- query identifiability is failed by the registered identifiability reason, or
  passed when the underlying update was admissible; and
- the physical guard is failed by the registered implausible-update reason, or
  passed when the underlying update was admissible.

Gauge/common-mode consistency, covariance calibration, material identity, and
robust-support sufficiency remain `null` unless an independently owned source or
calibration record supplies them. Supplied signals may fill unknown fields but
cannot contradict immutable update evidence. Adapter-owned metric and metadata
fields cannot be replaced.

Every generated record binds the claim-bearing update ID, admission ID,
inference-result ID, observation and linearization artifact IDs, provider
manifest ID, calibration artifact IDs, independently verified runtime source,
and complete strict-admission certificate. Mixed provider-manifest identities
and duplicate case IDs are rejected.

## Command

```bash
bpt diagnostic run diagnose-provider-failures \
  source-provider-evidence.json \
  source-provider-failure-report.json
```

The input reader requires one unchanged ordinary UTF-8 JSON file, applies a
finite 64 MiB default budget, rejects duplicate keys and non-finite constants,
and records the exact raw byte digest. The output is written atomically and
refuses to overwrite an existing path by default. Use `--overwrite` only for a
deliberately non-claim-bearing local rerun.

The report includes:

- primary and any-cause counts with equal case weight;
- accepted, classified-rejection, and unresolved-rejection totals;
- per-case explicit failures, reason-derived evidence, and unresolved signals;
- a canonical input-content digest and portable report identifier;
- a host-local `status_sha256` that additionally binds the exact input file path,
  byte count, and raw input-file digest written into the output; and
- the fixed taxonomy, precedence, and information boundary.

The diagnostic should be run before proposing another provider variant. A new
model component is justified only when the report localizes a failure that the
component can test on a separately frozen cohort.
