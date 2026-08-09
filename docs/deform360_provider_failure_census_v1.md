# Deform360 source-only provider failure census v1

This workflow turns already-frozen Deform360 source/provider decisions into one
object- or session-weighted failure census. It is the next diagnostic before a
new Prob4D provider, calibration method, physical query, or BayesianPhysTwin
update is proposed.

It deliberately does **not** rerun a target sweep. The completed official-Hub
provider remains terminal at its frozen support boundary, and the twelve
confirmation objects remain closed.

## Runner and storage boundary

The manual scientific job runs only on the sole runner selected by the
`self-hosted` label and verifies that `RUNNER_NAME` is `workstation2`. It binds:

- storage root: `/mnt/lexar4tb/datasets/deform360`;
- official/raw revision: `data-7fea8e2`; and
- adaptive-confirmation download:
  `adaptive-confirmation-download-5a9c56d593462486bdd0953dcaf6f9c643bf8370`.

The two raw roots are registered and checked as directories, but they are never
recursively listed, hashed, copied, modified, or passed to a payload-opening
command. The input must be one ordinary JSON file below the existing
`results/` tree. Output is a compact, no-overwrite directory below
`results/bayesian-phystwin/deform360-provider-failure-census-v1/`.

## Required input

Prepare the strict
[`provider_failure_evidence` v1](provider_failure_decomposition.md) JSON with one
case per registered physical object or acquisition session. Do not count frames,
points, tracks, views, rows, or taxels as independent cases.

The metadata must explicitly contain:

```json
{
  "split": "source-only",
  "statistical_unit": "physical-object",
  "confirmation_payloads_opened": false,
  "adaptive_confirmation_payloads_opened": false,
  "target_outcomes_used": false,
  "future_frames_used": false,
  "replacement_allowed": false
}
```

The dispatch also requires the lowercase SHA-256 of the exact JSON bytes. The
workflow rejects path normalization, symbolic-link resolution, files outside the
results tree, a digest mismatch, and inputs larger than 64 MiB.

## Dispatch

After this workflow is merged to protected `main`, open **Actions → Deform360
source-only provider failure census v1 → Run workflow** and provide:

- `execute_authorized = true`;
- `evidence_relative_path`, relative to the Deform360 `results/` directory; and
- `expected_input_sha256`, the exact lowercase SHA-256 of that file.

Pull requests execute only hosted contract tests. They cannot schedule the
self-hosted payload job.

## Outputs

The uploaded compact artifact contains:

- `provider-failure-report.json`, the versioned equal-case decomposition;
- `command-summary.json`, the installed CLI receipt;
- `summary.md`, a human-readable primary-category census;
- `execution-receipt.json`, binding revision, run, runner, input digest, report
  identity, counts, and the closed information boundary; and
- `SHA256SUMS`.

A rejection is retained as evidence. An unresolved rejection stays unresolved;
the workflow does not guess a favorable cause or change the original decision.

## How the census guides the next experiment

Use the dominant, independently weighted failure category to choose one narrow
next hypothesis:

| Dominant category | Defensible next source-only experiment |
| --- | --- |
| Unsupported provider geometry | Test the separately versioned object-level joint-sparse multi-view route; do not delete failed cameras from the frozen method. |
| Numerical non-convergence | Repair or reject the numerical path before spending GPU time on another provider. |
| Unidentifiable physical query | Redesign anchors or query support on a new source split rather than relaxing the target guard. |
| Coherent gauge/common-mode bias | Add independently grounded metric evidence or a genuinely new provider with complete joint gauge uncertainty. |
| Covariance miscalibration | Fit and validate covariance only on source/calibration objects, with object-balanced folds. |
| Association/material identity failure | Freeze a new persistent-tracklet or material-identity model before fresh outcomes. |
| Outlier-dominated evidence | Test one preregistered robust-support change while retaining exact fallback. |
| Physical model/readout mismatch | Compare a bounded discrepancy/readout correction against the unchanged physical fallback. |
| Technical failure | Repair execution and rerun the identical source-only contract; do not treat it as scientific support. |

No category authorizes adaptive-confirmation access, confirmation opening, target
outcome use, or promotion of the current provider.
