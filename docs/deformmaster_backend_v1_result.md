# DeformMaster causal-ingestion result v1

## Decision

**Portable adapter gate passed; public-release causal producer gate failed.**

Bayesian-PhysTwin can now ingest a fixed-identity DeformMaster surface rollout
when an external producer proves a prefix-only information boundary. The
current public release cannot honestly create that attestation, so empirical
advancement remains unevaluated and no target run is authorized.

## Source-only audit

- upstream repository: `CAN-Lee/DeformMaster`
- audited commit: `c7b3510a38b3fccbfe12cc6557aaf58d9ea823dc`
- dataset loader SHA-256:
  `ddb9c612bcd99f37986585a2defeb5cc51fe071c77193333ea56c203dd7b99b4`
- playground engine SHA-256:
  `8849908e4396baa741e5b0b3310ec7e026e2cff5844f0eae7127a412e2c01ae2`
- README SHA-256:
  `1a2797c3689b2f42657de01ee8bc702378b47863009b3fb210381c6f2392c45a`

The gate failed for three independently sufficient reasons:

1. `dataset_mpm.py` constructs router tracks from the complete loaded
   `object_points` sequence rather than a declared observation prefix.
2. `playground_engine.py` computes `auto_offset` from flattened
   `gt_surface_tracks` across all loaded frames.
3. The release explicitly omits full training code and does not supply the
   checkpoint training-object provenance required to prove target exclusion.

These are source-interface findings. No future trajectory, benchmark outcome,
or target metric was opened.

## Adapter controls

The new contract rejects:

- router, initialization, or offset ranges crossing the prefix boundary;
- any future object-track, RGB, depth, or outcome access flag;
- a target object present in checkpoint training data;
- changed checkpoint, configuration, training-manifest, or raw-rollout bytes;
- inconsistent frame-zero material identity; and
- mutated portable output files.

It also verifies byte-identical materialization for fixed inputs and validates
the resulting physical archive without importing DeformMaster.

Verification passed 109 shared DeformMaster, MatPhys, Newton, command-registry,
and distribution-contract tests. Changed-file Ruff and focused strict MyPy also
passed. Wheel and source-distribution builds, Twine checks, and an installed
base-wheel import without importing DeformMaster also passed. Focused coverage
was 98% for the backend module and 96% for its CLI.

## Next admissible step

Implement a small upstream-compatible causal exporter that:

1. slices dynamic router tracks to the permitted prefix;
2. computes initialization and frame placement from frame zero or the prefix;
3. accepts the known future controller action without reading future object
   observations;
4. binds checkpoint training data and target exclusion; and
5. seals its raw prediction before source-case future scoring.

Only then should one already-open source case compare DeformMaster against the
incumbent physical backend with exact fallback. The released all-frame path
must not be relabeled as a predictive benchmark.
