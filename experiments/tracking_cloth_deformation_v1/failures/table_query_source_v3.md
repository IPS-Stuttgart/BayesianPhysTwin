# Tracking Cloth table-query source v3: technical failure record

Status: **incomplete technical execution; no scientific result; reserved target remained closed**.

## Execution identity

- GitHub Actions run: [`33324078551`](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/33324078551)
- Workflow: `Tracking Cloth table-query source feasibility v3`
- Source job: `99291105124`
- Runner labels: `[self-hosted, Linux, X64, gpuserver4090]`
- Source branch: `science/tracking-cloth-table-query-source-v3`
- Evaluated revision: `4f61e610d50e9c446b6039a3ff5318371ea622a0`
- Execution date: 2026-08-30
- Retained artifact: `tracking-cloth-table-query-source-v3-33324078551`
- Artifact ID: `9735787762`
- Artifact digest: `sha256:500664d3c369283f94ba42e85bdb8f2e31fb3602d990bb45b775f86bfa9e1d16`

The hosted validation and request-authorization jobs passed. The self-hosted source job failed during construction of the first real-data causal input.

## Intended source-only protocol

The v3 protocol attempted a target-closed table-collision feasibility study using the public Tracking Cloth Deformation dataset. It registered:

- `half_lay_low_friction` and `half_lay_high_friction` as source/probe actions;
- `full_lay_low_friction` and `full_lay_high_friction` as prospective target queries;
- the repository strict CSV parser's dense 120 Hz segment;
- a 0.5-second all-marker causal prefix and 3.5-second forecast horizon;
- an 81-member stiffness, damping, and surface-friction model bank;
- fixed-order, parameter-information, and task-directed probe policies; and
- `target_scoring_authorized=false` and `paper_claim_authorized=false`.

The protocol required the first frame of each strict dense segment to supply metric scale and the 20-marker layout. No post-prefix `full_lay` free-marker coordinate was permitted to enter source fitting or probe selection.

## Exact failure

The source job terminated with:

```text
ValueError: setup frame is incomplete in cotton_A2_half_lay_low_friction.csv
```

The exception originated in `table_query_source_v2.input_view`, called by the v3 strict-segment wrapper. The real recording's first strict dense frame contains at least one missing marker, so it cannot satisfy the v3 requirement of a complete 20-marker frame for scale and layout initialization.

The implementation failed closed rather than dropping the recording, remapping markers using later outcomes, or silently imputing the setup geometry.

## Consequences

Because initialization failed before model fitting completed, this run produced no valid:

- 81-member source posterior;
- task-directed or parameter-information probe choice;
- policy-disagreement count;
- source cross-validation RMSE comparison;
- source-gate pass/fail decision; or
- collision-query scientific conclusion.

This is therefore **not a negative result for query-directed probing**. It is a technical support failure caused by an overly strict initialization assumption.

The reserved `full_lay` target remained closed: no reserved post-prefix free-marker outcome was scored, target scoring was not authorized, and no paper claim was authorized.

## Required remediation

Any retry must use a new protocol version rather than reinterpret v3. A defensible repair should:

1. select the earliest complete frame inside a prospectively fixed causal initialization window;
2. apply the same completion deadline and selection rule to every recording;
3. bind the selected frame index and missingness audit into the source artifact;
4. reject records that lack a complete frame within the frozen window rather than dropping them after inspection;
5. avoid interpolation or marker remapping that uses future free-marker outcomes; and
6. preserve the existing target-closure and claim-authorization boundaries.

The v3 run and this failure record should remain immutable provenance for the abandoned first-frame-completeness contract.
