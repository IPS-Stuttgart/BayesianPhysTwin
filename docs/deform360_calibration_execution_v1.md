# Deform360 calibration execution seal v1

## Purpose

The official-Hub Deform360 visuotactile protocol already fixes a fresh
metadata-only cohort, one exact Prob4D/MotionCrafter producer, and a finite-group
calibration design. The remaining pre-confirmation task is to prove that every
calibration-derived choice was assembled from the ten locked calibration
objects before any of the twelve confirmation payloads was opened.

`bayesian_phystwin.deform360_calibration_execution` and the grouped command

```text
bpt experiment run seal-deform360-calibration
```

provide that bridge. They do not fit a model or download data. They validate,
copy, content-address, and atomically publish the outputs of a completed
calibration execution.

## Required inputs

The command requires:

- the committed Stage-0 selection lock;
- one `Deform360VisualProviderLockV1`;
- one `EvidenceUseLedgerV1` for the calibration cohort;
- exactly one `Deform360CalibrationArtifactRefV1` for each of the eight
  registered calibration roles;
- the exact clean BayesianPhysTwin implementation revision; and
- an explicit acknowledgement that calibration payloads, but no confirmation
  payloads or target outcomes, were opened.

The evidence ledger must use
`case_id=deform360-official-hub-calibration-cohort-v1`. Every entry must have
`inference_role=calibration_only`, identify one or more Stage-0 calibration
objects in metadata, and collectively cover all ten independent objects.
Evidence from a confirmation object fails closed.

## Eight roles and four Stage-1 identities

The full calibration bundle retains all eight roles:

1. `contact_feature_and_grouping`;
2. `contact_linearization_and_covariance`;
3. `anchor_bias_prior`;
4. `visual_reliability_and_gauge`;
5. `normalized_evidence`;
6. `physical_response_and_closure`;
7. `regret_guard`; and
8. `conformal_interval`.

The narrower `Deform360VisualCalibrationLockV1` has four component identities.
The execution builder derives those identities deterministically from the
content-addressed role references:

- visual: roles 4 and 5;
- contact anchor: roles 1, 2, and 3;
- guard and closure: roles 6 and 7; and
- interval: role 8.

Consequently, changing any selected candidate, source file, calibration group,
implementation revision, or role reference changes the Stage-1 lock, the full
bundle, and the final execution seal.

## Atomic publication

A successful command creates one new directory containing:

```text
visual-calibration-lock.json
calibration-bundle.json
calibration-execution-seal.json
summary.json
STATUS.md
SHA256SUMS
sources/
```

The `sources/` tree preserves the exact Stage-0 lock, visual-provider lock,
evidence ledger, eight calibration references, registered protocol amendments,
and implementation files. Every source is copied byte-for-byte and recorded in
the bundle. Symlinks, missing roles, duplicate paths, a dirty checkout, a
revision mismatch, an existing output directory, and output inside the Git
checkout are rejected. Logical source names must already be canonical relative
POSIX paths; spellings with `./`, redundant separators, traversal, or Windows
separators are not normalized into accepted identities.

All files are written in a temporary sibling directory. The complete directory
is atomically renamed only after nested contract validation, independent
cross-artifact verification, and checksum generation succeed.

## Example

```bash
bpt experiment run seal-deform360-calibration \
  --selection-lock \
    protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-lock /sealed/visual-provider-lock.json \
  --evidence-ledger /calibration/evidence-use-ledger.json \
  --artifact contact_feature_and_grouping=/calibration/contact-features.json \
  --artifact \
    contact_linearization_and_covariance=/calibration/contact-model.json \
  --artifact anchor_bias_prior=/calibration/anchor-bias.json \
  --artifact \
    visual_reliability_and_gauge=/calibration/visual-calibration.json \
  --artifact normalized_evidence=/calibration/evidence-scaling.json \
  --artifact \
    physical_response_and_closure=/calibration/physical-closure.json \
  --artifact regret_guard=/calibration/regret-guard.json \
  --artifact conformal_interval=/calibration/conformal-interval.json \
  --implementation-revision "$(git rev-parse HEAD)" \
  --repository-root "$PWD" \
  --output-dir /sealed/deform360-calibration-v1 \
  --calibration-payloads-opened
```

The emitted `confirmation_opening_token` is the only token authorized for the
subsequent confirmation runner. Any changed cohort, provider, calibration
choice, evidence ledger, source bytes, or implementation revision produces a
different token.

## Information and claim boundary

A valid execution seal proves that the declared calibration choices, source
bytes, cohort, and information boundary are mutually consistent under the
implemented contracts. It is infrastructure and calibration-lineage evidence,
not an empirical result. It does not establish provider competence, tactile
benefit, physical-query improvement, predictive calibration, deployment safety,
Causal4D intervention benefit, or state of the art.
