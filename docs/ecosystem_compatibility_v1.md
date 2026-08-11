# Ecosystem compatibility table v1

## Purpose

BayesianPhysTwin, Prob4D, and Causal4D evolve in separate repositories but share
versioned observation, belief, replay, and artifact contracts. The installed
resource

```text
bayesian_phystwin/contract_data/ecosystem_compatibility_v1/table.json
```

is the normative development-compatibility table for the current package lines.
`bayesian_phystwin.ecosystem_compatibility_v1` validates the resource strictly,
makes it recursively immutable, and assigns a canonical SHA-256 `table_id`.

The table is **not** an experiment revision lock. A compatible package range says
that the public interfaces are expected to interoperate. Claim-bearing evidence
must still bind exact repository revisions, installed artifacts, dependency
resolver input, and content digests.

## Supported package lines

| Component | Distribution range | Python | Required numerical dependencies |
| --- | --- | --- | --- |
| BayesianPhysTwin | `bayesian-phystwin>=0.4,<0.5` | `>=3.10` | `numpy>=1.23` |
| Prob4D | `prob4d>=0.4,<0.5` | `>=3.10` | `numpy>=1.24` |
| Causal4D | `causal4d>=0.5,<0.6` | `>=3.10` | `numpy>=1.24`, `packaging>=23`, `scipy>=1.10` |

These are development ranges. They do not replace exact source revisions in a
`RunManifestV2`, provider attestation, experiment protocol, or released evidence
bundle.

## Prob4D to BayesianPhysTwin

### Provider v1: frozen compatibility

`prob4d.provider_v1` remains available for historical reproduction and
exploratory interoperability. Its provider API version is 1. The required
cross-repository schemas are:

| Artifact | Supported schema versions |
| --- | --- |
| `GaugeCovarianceCalibrationV1` | 1 |
| `MetricGaugeAnchor` | 1 |
| `ObservationBeliefV1` | 1 |
| `ObservationFactorBundle` | 3 |
| `PointUncertaintyCalibrationV1` | 1 |
| `Prob4DCausalObservationStream` | 2 |

This row is not a claim-bearing admission boundary. The content-addressed
provider-v1 manifest deliberately retains the historical source-repository
identity `FlorianPfaff/Prob4D`; changing that field would change historical
manifest identities.

### Provider v2: claim-bearing admission

`prob4d.provider_v2` is the supported claim-bearing producer boundary. Its
provider API version is 2. BayesianPhysTwin independently validates the complete
provider manifest and attestation without importing Prob4D.

| Artifact | Supported schema versions |
| --- | --- |
| `ProviderAttestation` | 1 |
| `ObservationBeliefV1` | 1 |
| `ObservationFactorBundle` | 4 |
| `ObservationFactorStreamV1` | 1 |
| `Prob4DCausalObservationStream` | 2 |
| `MetricGaugeAnchor` | 1 |
| `GaugeCovarianceCalibrationV1` | 1 |
| `PointUncertaintyCalibrationV1` | 1 |

Claim-bearing admission additionally requires complete calibration metadata,
an independently matched runtime revision, provider-v2 capabilities, and the
registered covariance semantics. Schema-v2/v3 factor bundles remain
conservative marginal-only compatibility inputs; provider-v2 joint
cross-window factors require schema 4.

## BayesianPhysTwin to Causal4D

Causal4D `0.5.x` accepts BayesianPhysTwin `0.4.x` through versioned public
modules only. The machine-readable table mirrors Causal4D's provider registry:

| Module | API | Lifecycle | Role |
| --- | ---: | --- | --- |
| `bayesian_phystwin.causal4d_artifacts_v1` | 1 | frozen compatibility | released artifact I/O |
| `bayesian_phystwin.causal4d_artifacts_v2` | 2 | production additive | released visual artifact I/O |
| `bayesian_phystwin.causal4d_belief_provider_v1` | 1 | frozen compatibility | fixed-anchor endpoint inference |
| `bayesian_phystwin.causal4d_belief_provider_v2` | 2 | additive development | model-averaged horizon discrepancy |
| `bayesian_phystwin.causal4d_graph_provider_v1` | 1 | production | spring graph and controller grouping |
| `bayesian_phystwin.causal4d_provider_v1` | 1 | frozen compatibility | scientific compatibility facade |
| `bayesian_phystwin.causal4d_provider_v2` | 2 | production | request-complete replay |
| `bayesian_phystwin.causal4d_public_provider_v1` | 1 | diagnostic | source-locked public diagnostics |
| `bayesian_phystwin.causal4d_tree_block_provider_v1` | 1 | production additive | claim-bearing tree-block query covariance |

The common required artifact schemas are version 1 for `GraphBelief`,
`TwinBelief`, `FixedBayesianAnchorConfig`, `RobustEndpointPosterior`,
`ReplayRequest`, `ReplayTrajectory`, `ScheduledContactReplayRequest`, and
`ScheduledContactReplayResult`. Provider manifests and the matching local
Causal4D contract remain authoritative for module-specific capabilities and
additional schemas.

## Reading the installed table

```python
from bayesian_phystwin.ecosystem_compatibility_v1 import (
    load_ecosystem_compatibility_table_v1,
)

table = load_ecosystem_compatibility_table_v1()
print(table.table_id)
print(table.component("prob4d")["supported_versions"])
print(
    table.interface("prob4d-provider-v2-to-bayesian-phystwin")[
        "required_artifact_schema_versions"
    ]
)
```

The loader rejects duplicate JSON keys, non-finite values, unknown fields,
noncanonical package ranges, reordered or duplicate provider rows, unsupported
schema-version lists, coerced Boolean flags, and any weakening of the evidence
boundary.

## Continuous compatibility ownership

The contract test `tests/test_ecosystem_compatibility_v1.py` belongs permanently
to the centralized `stable-core-coverage`, `core-contracts`, and
`provider-contract` suites. This makes table, package-range, provider-module,
schema-version, packaging, installed-resource, immutability, and evidence-boundary
drift fail in every relevant repository gate rather than only in the full suite.

## Evidence boundary

A matching table row establishes only contract-level interoperability:

- development ranges are not evidence locks;
- exact repository revisions and artifact digests remain mandatory for
  claim-bearing runs;
- an installed-wheel golden path is compatibility evidence, not accuracy or
  uncertainty-calibration evidence;
- provider competence must be measured separately from downstream
  physical-query or intervention benefit; and
- compatibility does not establish unseen-object transfer, calibrated raw
  covariance, Causal4D physical evidence, deployment authorization, or state of
  the art.
