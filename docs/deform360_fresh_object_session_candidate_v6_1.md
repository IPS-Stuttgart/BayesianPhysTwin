# Deform360 v6.1 source-prefix candidate producer

## Purpose

This stage makes the registered v6.1 `D1` dynamic endpoint model average
executable on the sealed public Deform360 real-world recordings. No new
physical measurement or human approval is required. The producer consumes only
the causal source prefix and the already sealed physical, persistence, and
MotionCrafter disjoint-baseline visual products.

The stage is deliberately separated from source-suffix scoring. Its only valid
success artifact is a content-addressed 100-record raw nested candidate panel
plus a protected-run execution receipt. A technical failure produces a terminal
retained-failure receipt and does not authorize replacement.

## Fixed candidate set

- `B0`: exact physical-fallback trajectory from the sealed upstream artifact;
- `B1`: exact last-causal-residual trajectory from the sealed upstream artifact;
- `D1`: physical prediction plus a causal residual trajectory from the frozen
  dynamic endpoint model average;
- `VT1-working`, `VT1-observed`, and `VT1-sandwich`: explicitly unavailable,
  because the public release does not identify tactile channels with robot
  axes.

`D1` fits no parameter across source objects. Its model components and
hyperparameters are frozen, and `evidence_pooling="object"` pools graph-node
evidence only within the target object's causal prefix. The eight-object nested
fit roster is retained solely as the registered gate's eligibility and
provenance structure.

The sealed visual bundle was produced by the Prob4D pipeline, but this stage
consumes `baseline_disjoint.npz`, not Prob4D decoded-uniform overlap fusion.
The upstream source-plan field remains named `decoded_uniform` for schema
compatibility; the bound paths and v6.1 provenance identify the product
unambiguously. The producer runs no new MotionCrafter or Prob4D inference and
does not relabel visual observations as tactile evidence.

## Observation update

Geometry determines soft four-nearest-node association probabilities. Prior
perception reliability is computed separately from source confidence, mask
distance, and overlap disagreement; it does not depend on the innovation
against the PhysTwin state. The sealed disjoint product does not supply overlap
disagreement, so that cue is identically neutral in this execution. The state
innovation is processed once by the dynamic robust filter.

Rows are aggregated once per frame and graph node. Duplicating a correlated
pixel block therefore cannot create additional filter updates. Unsupported
nodes remain unsupported: there is no nearest-node fill. `D1` covariance is the
native law-of-total-covariance output, expressed in square metres. Assignment
mixture spread remains part of observation uncertainty through the existing
prefix materialization contract.

## Frozen lineage

The amendment binds all of the following by content identity and file SHA-256:

- source execution lock;
- upstream source plan;
- upstream 100-record prediction batch;
- upstream prediction-panel receipt;
- upstream protected execution receipt.

The runner rehashes every upstream source seal and physical prediction archive,
then publishes and independently reloads every candidate archive before the
panel receipt can exist.

## Information boundary

The producer may read frames in the causal range `[0, 58)`. It has no endpoint
planner or scorer and may not access the registered development suffix
`[58, 76)`. It does not authorize suffix scoring, the source gate, independent
confirmation, or a scientific claim. Those are separate, later decisions.

The one-shot workflow additionally requires protected `main`, the designated
`workstation2` runner, an empty durable output root, and an exclusive filesystem
claim. It uploads only compact pre-suffix receipts and hashes.

## Commands

Validate the frozen amendment:

```bash
python scripts/science/run_deform360_fresh_object_session_candidate_v6_1.py \
  validate-amendment \
  --amendment \
  protocols/amendments/deform360_official_hub_fresh_object_session_v6_candidate_producer.json
```

The protected producer is dispatched only through
`.github/workflows/deform360-v61-candidate-producer.yml`. Manual local execution
does not create authoritative empirical evidence.
