# Deform360 calibration observability report

## Purpose

The official-Hub Deform360 protocol fixes ten calibration objects before raw
payload access and keeps twelve separate confirmation objects sealed. The
low-level `compare_marginal_observability` diagnostic already measures how much
an independent contact family changes a nuisance-marginalized physical query.
This report adds the missing study-level boundary around that calculation.

A report contains exactly one result for every locked calibration object. It
compares:

1. the visual explicit-gauge reference belief after all declared camera and
   source nuisances are marginalized; and
2. the otherwise identical belief after adding the independently calibrated
   tactile or proprioceptive contact family and its own nuisance model.

The report is calibration evidence for Stage 1. It is not a confirmation result
and it does not authorize confirmation access by itself.

## Per-object case contract

`Deform360CalibrationObservabilityCaseV1` binds one exact object and episode to:

- the committed Stage-0 selection identity;
- the target-blind visual-provider lock identity;
- the successful calibration-source terminal-record identity;
- the exact analysis implementation revision;
- one content-addressed physical-query definition;
- exact visual-reference, visual-plus-contact, and contact-anchor artifact IDs;
- the two nuisance-marginalized physical-state precision matrices;
- the exact physical-query Jacobian;
- all source-file SHA-256 values; and
- a closed information boundary.

The numerical comparison is recomputed from the two precision matrices through
`compare_marginal_observability`. A supplied comparison cannot override the
recomputed mutual-information gain, weakest-direction precision ratio, variance
reduction, or rank changes.

A candidate that reduces query information fails closed. This normally indicates
mismatched priors, nuisance domains, evidence order, or query definitions rather
than valid negative contact information.

## Technical failures and support

Every one of the ten selected objects remains in the denominator. An object that
cannot produce a valid comparison is recorded as
`technical_failure_without_replacement` with a nonempty failure reason and no
numerical result.

The frozen source-support rule is reused without modification:

- at least 8 of 10 objects must be evaluated; and
- at least 4 of 5 objects in both the `sheet` and `volumetric` strata must be
  evaluated.

A completed report that misses this rule is a valid negative feasibility result.
It does not permit object replacement, threshold relaxation, or confirmation
access.

## Object-balanced summaries

`Deform360CalibrationObservabilityReportV1` reports equal-object summaries for:

- mutual-information gain in nats;
- weakest-direction precision ratio;
- mean marginal-variance reduction;
- numerical-rank gain; and
- effective-rank gain.

It also reports how many evaluated objects have a positive gain under one
explicit numerical tolerance. The report does not introduce a new scientific
acceptance threshold. The registered confirmation decision remains the
object-balanced physical-query loss and harmful-update rule in the main
Deform360 protocol.

## Source and implementation revisions

The calibration-source revision and the later observability-analysis revision
are retained separately. This matters because preparation can finish under one
reviewed source revision while the report is assembled by a later compatible
implementation. Replacing either revision changes the report identity.

The claim-bearing path requires a fully validated successful calibration-source
terminal record. It independently requires:

```text
status = succeeded
exit_code = 0
confirmation_boundary_verified = true
confirmation_payloads_opened = false
support_gate.support_passed = true
```

The record's Stage-0 selection and visual-provider lock must exactly match the
ones used by the report.

## Building the report

First publish ten strict case JSON files. Then run:

```bash
python scripts/science/build_deform360_calibration_observability_report.py \
  --selection-lock \
    protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --stage0-protocol \
    protocols/deform360_official_hub_visuotactile_v1.json \
  --visual-provider-lock /sealed/visual-provider-lock.json \
  --calibration-source-run-record /sealed/execution-manifest.json \
  --case /calibration/observability/object-01.json \
  --case /calibration/observability/object-02.json \
  --case /calibration/observability/object-03.json \
  --case /calibration/observability/object-04.json \
  --case /calibration/observability/object-05.json \
  --case /calibration/observability/object-06.json \
  --case /calibration/observability/object-07.json \
  --case /calibration/observability/object-08.json \
  --case /calibration/observability/object-09.json \
  --case /calibration/observability/object-10.json \
  --implementation-revision "$(git rev-parse HEAD)" \
  --physical-query-id "<content-addressed query ID>" \
  --output /sealed/deform360-calibration-observability-v1.json
```

Exit code `0` means the frozen 8/10 and 4/5 source-support gate passed. Exit code
`3` means the report completed but support was insufficient. Exit code `2`
means a contract or provenance failure.

## Stage-1 use

The report ID and exact source-file digest can be referenced by the
`contact_linearization_and_covariance`, `anchor_bias_prior`, or
`physical_response_and_closure` calibration-selection evidence as appropriate.
The complete eight-role calibration bundle and evidence-use ledger still have to
be sealed through `bpt experiment run seal-deform360-calibration` before the
confirmation-opening token exists.

## Information and claim boundary

A valid report proves that the declared ten-object observability calculations,
technical failures, source identities, and information boundary are internally
consistent. It does not establish provider competence, tactile benefit,
physical-query accuracy, predictive calibration, Causal4D intervention benefit,
deployment safety, or state of the art. All twelve confirmation payloads and all
target outcomes remain forbidden while this report is built.
