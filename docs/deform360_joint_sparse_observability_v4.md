# Deform360 joint-sparse observability v4

## Purpose

Version 4 is a new, development-only Deform360 protocol. It does not modify or
reinterpret the frozen v1, v2, or v3 results. Those versions established that
complete per-camera support and then per-window dense spatial support were not
available for the released causal prefixes.

The v4 question is narrower and structurally different:

> Can complementary partial factors from multiple cameras and causal windows
> jointly identify the preregistered physical-query subspace after gauge and
> bias nuisance variables are marginalized?

A camera is no longer required to be independently sufficient. Every valid
partial factor is retained, and unsupported factors contribute no likelihood
term. Admission occurs at the physical-object/query level.

## Information boundary

The protocol consumes only structural uncertainty information:

- conditional 3-D observation covariance;
- state and local gauge Jacobians;
- the causal tree-sparse gauge prior;
- camera, causal-window, spatial-cluster, and dependence-group identities;
- source reliability, association probability, and group likelihood power;
- optional shared and view-specific bias designs; and
- one frozen physical-query Jacobian per object.

It explicitly excludes predicted point values, residuals, calibration outcomes,
future frames, confirmation payloads, adaptive-confirmation payloads, target
outcomes, replacements, and human selection. The ten previously opened source
objects are development objects only. A v4 development pass does not authorize
confirmation access.

## Joint information calculation

For factor `i`, the local linear design is

```text
H_i = [H_state,i, H_gauge,i, H_shared,i, H_view,i].
```

The gauge prior is assembled in precision form from the registered causal-tree
transition and innovation factors. Observation rows are whitened by their
conditional covariance. Dependence groups receive a bounded effective-sample
mass, preventing many correlated rows from manufacturing arbitrary precision.

After accumulating the joint information matrix, v4 marginalizes gauge and bias
nuisance variables with a Schur complement. For physical-query Jacobian `J_q`,
only combinations orthogonal to the state-information nullspace are assigned
finite precision. The evaluator reports:

- state and query rank;
- the complete query-precision eigenspectrum;
- weakest query precision and condition number;
- query-nullspace overlap;
- the information fraction attributable to each camera;
- leave-one-camera-out query-rank retention; and
- leave-one-window-out query-rank retention.

## Frozen development gate

The policy in
`protocols/locks/deform360_official_hub_joint_sparse_observability_v4.json`
requires, per object:

- at least two cameras and two causal windows;
- at least eight distinct spatial clusters jointly;
- full registered-query rank;
- weakest query precision at least `1e-9`;
- query condition number at most `1e10`;
- no camera contributing more than `85%` of total query precision;
- at least `75%` query-rank retention after removing any one camera; and
- at least `75%` query-rank retention after removing any one window.

The development report requires at least eight supported objects, including at
least four sheet and four volumetric objects, with no technical replacement.
These thresholds are frozen before any adaptive-confirmation or confirmation
payload is opened. They are design diagnostics, not empirical calibration or
benefit claims.

## Relation to tree-sparse Prob4D

`build_deform360_joint_sparse_factor_batch_from_tree_sparse_v4()` consumes the
existing claim-bearing tree-sparse Prob4D adapter. It reuses conditional
covariance, local gauge Jacobians, tree transitions, row powers, and evidence
lineage without reading innovations or residuals. This avoids a parallel factor
format and avoids dense gauge-covariance materialization.

## Portable manifest

The evaluator consumes a content-addressed manifest with one descriptor/NPZ pair
per development object. Each descriptor is the exact
`Deform360JointSparseFactorBatchV4.identity_record()` plus `input_id`. The NPZ
must contain exactly:

```text
observation_covariance_m2
state_jacobian
local_gauge_jacobian
gauge_indices
parent_indices
transition_matrices
innovation_scale_tril
query_jacobian
prior_reliability
association_probability
composite_weight
shared_bias_jacobian
view_bias_jacobian
```

Every manifest member is bound by path, byte count, SHA-256, and content ID. The
CLI publishes policy, manifest, one result per object, an object-balanced report,
and recursive checksums atomically. A support-negative report is a complete
result and is returned with exit code `3`; it must not be rescued by replacing
objects, cameras, factors, or thresholds.

## Execution

```bash
python scripts/science/evaluate_deform360_joint_sparse_observability_v4.py \
  --policy protocols/locks/deform360_official_hub_joint_sparse_observability_v4.json \
  --manifest /path/to/development-manifest.json \
  --output-dir /path/to/new-output-directory
```

The guarded GitHub workflow runs the same evaluator on the sole `self-hosted`
runner only after manual authorization from protected `main`. It accepts a
manifest below the Deform360 results tree and verifies that neither the official
raw revision nor the adaptive-confirmation tree is traversed by this stage.

## Decision after v4

A development pass permits only a separately reviewed freeze of a fresh,
independent calibration protocol. It does not permit use of the ten development
objects for a claim-bearing calibration and does not open the twelve-object
confirmation set. If joint object-level observability remains inadequate, the
visual route should stop rather than relax the gate post hoc.
