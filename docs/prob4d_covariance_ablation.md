# Controlled Prob4D covariance ablation

`bpt diagnostic run audit-prob4d-covariance-ablation` verifies and summarizes a
matched five-way covariance ablation for the Prob4D-to-BayesianPhysTwin update.
It is designed to answer one narrow question:

> What changes when the observation mean, physical model, scored units, fallback,
> risk policy, calibration split, and software stack are held fixed, while only
> the Prob4D covariance treatment changes?

The command reuses the stable decisive-evidence analyzer for operational loss,
exact fallback, harmful accepted updates, threshold-native risk–coverage,
matched-count risk–coverage, horizon conditioning, reliability conditioning,
identifiable-rank conditioning, and interval calibration.

## Required variants

Every input must contain each treatment exactly once:

| Treatment | Shared uncertainty | Explicit gauge factors |
| --- | --- | --- |
| `full_joint` | reported at scale `1` | enabled |
| `block_diagonal` | within-block covariance retained | disabled |
| `independent_rows` | row dependence removed | disabled |
| `shared_uncertainty_removed` | shared contribution set to `0` | disabled |
| `shared_uncertainty_underreported` | fixed scale strictly inside `(0, 1)` | enabled |

`full_joint` is the candidate whose contribution is attributed against every
other treatment. `reference_treatment` selects one ablated comparator for the
complete decisive-evidence summary and must therefore not be `full_joint`.

## Isolation contract

Each variant binds a distinct run manifest and covariance artifact. The following
SHA-256 identities must be byte-for-byte identical across all five variants:

- observation means;
- scored statistical units;
- physical linearizations;
- exact fallback policy;
- risk and admission policy;
- calibration partition; and
- software stack.

The top-level lock must also state that source or calibration policy was frozen
and that the only permitted difference is
`prob4d-covariance-treatment-only`. A changed invariant, missing treatment,
duplicate method, duplicate run manifest, mismatched evidence method, or altered
statistical unit is rejected rather than interpreted.

These digests should be built from portable content-addressed manifests, not
host-local file names or timestamps.

## Input

The input is one strict UTF-8 JSON object. Duplicate keys, non-finite constants,
nonordinary files, files that change while being read, and inputs over the
64 MiB default budget are rejected.

```json
{
  "schema": "bayesian_phystwin.prob4d_covariance_ablation",
  "schema_version": 1,
  "ablation_id": "fresh-object-prob4d-covariance-v1",
  "reference_treatment": "independent_rows",
  "locked_factors": {
    "dataset_id": "fresh-object-panel-v1",
    "split_id": "confirmation-v1",
    "registered_statistical_unit": "physical object",
    "source_or_calibration_policy_frozen": true,
    "allowed_variant_difference": "prob4d-covariance-treatment-only"
  },
  "variants": [
    {
      "method": "prob4d-full-joint",
      "treatment": "full_joint",
      "shared_uncertainty_scale": 1.0,
      "gauge_factors_enabled": true,
      "run_manifest_sha256": "<64 lowercase hexadecimal characters>",
      "covariance_artifact_sha256": "<64 lowercase hexadecimal characters>",
      "observation_mean_sha256": "<common digest>",
      "scored_units_sha256": "<common digest>",
      "physical_linearization_sha256": "<common digest>",
      "fallback_policy_sha256": "<common digest>",
      "risk_policy_sha256": "<common digest>",
      "calibration_partition_sha256": "<common digest>",
      "software_stack_sha256": "<common digest>"
    }
  ],
  "evidence": {
    "contract": "bayesian-phystwin-decisive-evidence-v1",
    "schema_version": 1,
    "protocol_id": "fresh-object-prob4d-covariance-v1",
    "statistical_unit": "physical object",
    "claim_boundary": "controlled covariance attribution only",
    "reference_method": "prob4d-independent-rows",
    "records": []
  }
}
```

The abbreviated example shows one variant only. A valid artifact requires all
five variants and a nonempty matched decisive-evidence record set. Every method
must occur on every metric and statistical unit, and every method must use the
same fallback loss for that metric and unit.

## Command

```bash
bpt diagnostic run audit-prob4d-covariance-ablation \
  runs/prob4d_covariance/input.json \
  runs/prob4d_covariance/report.json
```

Publication is atomic and refuses to replace an existing report. Use
`--overwrite` only for a deliberately non-claim-bearing local rerun.

## Output

The report contains:

- canonical input, locked-factor, and decisive-evidence content identities;
- the common invariant digests and all five run/covariance identities;
- the full decisive-evidence summary against the selected ablated reference;
- pairwise raw and operational attribution of `full_joint` against each other
  covariance treatment for every metric;
- a portable `report_id` over the complete report; and
- a host-local `status_sha256` that additionally binds the exact input file
  path, byte count, and raw-byte digest.

The report always sets `claim_authorized` to `false`. It is an attribution
diagnostic, not an automatic promotion gate.

## Experimental use

For physical evidence, freeze the provider, five covariance constructions,
source/calibration-derived admission thresholds, statistical unit, and all
invariant manifests before opening confirmation outcomes. Use distinct unopened
objects or acquisition sessions for confirmation; frames, tracks, views, points,
and taxels are not independent confirmation units merely because they produce
multiple records.

Compare the covariance variants to the strong last-residual baseline through the
existing `bpt evidence summarize` protocol. This diagnostic deliberately keeps
the covariance attribution internally controlled rather than mixing a
fundamentally different deterministic method into the five-way identity lock.

A positive controlled result does not establish real-provider competence,
calibrated raw posterior uncertainty, fresh-object physical benefit, Causal4D
intervention benefit, deployment safety, or overall state of the art. Those
claims require separately frozen prospective evidence.
