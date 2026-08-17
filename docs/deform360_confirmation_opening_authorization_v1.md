# Deform360 confirmation-opening authorization v1

## Purpose

The Stage-1 calibration bundle computes a deterministic confirmation-opening
token, but the token alone does not prove that the empirical calibration-source
run succeeded or that the preregistered ten-object contact-versus-visual
observability report passed its frozen support rule.

`Deform360ConfirmationOpeningAuthorizationV1` closes that gap. It is the
claim-bearing artifact that must exist before any of the twelve frozen
confirmation payloads are opened.

## Required evidence

The authorization binds all of the following in one content identity:

- the exact Stage-0 selection and its disjoint 10-calibration/12-confirmation
  physical-object cohorts;
- the target-blind visual-provider lock;
- one strictly validated calibration-source terminal record with
  `status=succeeded`, `exit_code=0`, a passed `8/10` plus `4/5`-per-stratum
  support gate, and an independently verified closed confirmation boundary;
- one supported object-balanced calibration observability report over the exact
  ten Stage-0 calibration objects;
- the exact file SHA-256 values of both the run record and observability report;
- three calibration roles whose `selection_evidence_id` is the observability
  report ID:
  - `contact_linearization_and_covariance`;
  - `anchor_bias_prior`; and
  - `physical_response_and_closure`;
- the complete eight-role calibration bundle and evidence-use ledger;
- the generated Stage-1 execution seal and confirmation-opening token; and
- explicit declarations that no confirmation payload or target outcome has been
  opened.

The observability report must itself retain the exact calibration-source
run-record bytes from which it was built. A matching content ID with different
serialized source bytes is rejected.

## Claim-bearing command

Use the additive sealing command from one clean reviewed checkout:

```bash
python scripts/science/seal_deform360_calibration_with_observability.py \
  --selection-lock \
    protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-lock /sealed/visual-provider-lock.json \
  --evidence-ledger /calibration/evidence-use-ledger.json \
  --artifact contact_feature_and_grouping=/calibration/contact-feature.json \
  --artifact contact_linearization_and_covariance=/calibration/contact-cov.json \
  --artifact anchor_bias_prior=/calibration/anchor-bias.json \
  --artifact visual_reliability_and_gauge=/calibration/visual-reliability.json \
  --artifact normalized_evidence=/calibration/normalized-evidence.json \
  --artifact physical_response_and_closure=/calibration/physical-response.json \
  --artifact regret_guard=/calibration/regret-guard.json \
  --artifact conformal_interval=/calibration/conformal-interval.json \
  --calibration-source-run-record /calibration/execution-manifest.json \
  --calibration-observability-report \
    /calibration/deform360-calibration-observability-v1.json \
  --implementation-revision "$(git rev-parse HEAD)" \
  --repository-root "$PWD" \
  --output-dir /sealed/deform360-stage1-authorized-v1 \
  --calibration-payloads-opened
```

The command first runs the existing Stage-1 sealer into a temporary directory.
It adds the run record, observability report, binding implementation, and command
source to the sealed source map. It then reloads every output, recomputes all
content identities, validates the complete binding, emits
`confirmation-opening-authorization.json`, rebuilds `SHA256SUMS`, and publishes
the requested directory atomically. A failed binding leaves no final output.

## Output boundary

A successful directory contains the ordinary Stage-1 artifacts plus:

- `confirmation-opening-authorization.json`;
- `confirmation-opening-summary.json`; and
- a `STATUS.md` section naming the authorization, run-record, and observability
  identities.

The confirmation runner must require the exact authorization ID and verify the
artifact before opening data. The lower-level bundle token remains useful as an
internal identity, but it is not sufficient claim-bearing authorization under
this protocol.

## Information and claim boundary

This artifact establishes information order, cohort continuity, source-byte
retention, support-gate completion, and exact cross-artifact identity only. It
does not establish provider competence, physical-query accuracy, tactile
benefit, predictive calibration, harmful-update risk below a chosen population
bound, Causal4D intervention benefit, deployment safety, or state of the art.
