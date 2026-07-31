# DEFORM DLO2 initialization amendment v1

## Reason

A target-free audit of the locked upstream DEFORM source found that the generic
`DEFORM_sim` constructor initializes every node count with the DLO1 rest
geometry. Upstream `train_DEFORM.py` replaces that state inside each DLO branch.
In particular, DLO2 uses its own 12-node rest geometry, recomputes edge and
region rest lengths, and uses DLO2-specific bend and twist stiffness values.

The prospective wrapper previously replaced only stiffness, using the DLO1
values for every DLO type, and did not replace the constructor's rest geometry.
That path was valid for the active DLO1 run but was not a faithful DLO2
construction.

## Information boundary

The mismatch was found before any DLO2 source trajectory or official DLO2
evaluation artifact was enumerated, hashed, loaded, or scored. The active DLO1
job runs from its unchanged archived commit and is not restarted. This amendment
therefore changes only unopened future DLO2 stages.

## Correction

The wrapper now parses the DLO-specific initialization directly from the
separately obtained, commit-locked upstream `train_DEFORM.py` using Python's AST.
It does not vendor upstream geometry or code. Before construction it verifies:

- the external checkout remains at commit
  `b73b8b8ecc033caefa693fab7898741d4e6dbeff` and tracked-clean;
- `train_DEFORM.py` has SHA-256
  `d45abe23a22b0f01fa266833844c4f9b71a2b7e375f8e955e3278b9e969acc55`;
- exactly one registered branch supplies the requested DLO initialization;
- the parsed node count matches the protocol;
- rest lengths are recomputed from the parsed transformed geometry; and
- every source checkpoint records the initialization contract and source hash.

Fresh DLO2, the all-56 refit, posterior reconstruction, and the one-shot
official evaluator all require `official-deform-dlo-initialization-v1`. Parent
protocol hashes are rebound after this amendment. No gate, split, checkpoint
schedule, posterior arm, evaluation metric, or acceptance threshold changes.

## Verification

Unit tests cover DLO1/DLO2 extraction, coordinate conversion, stiffness
separation, malformed and duplicate branches, node-count rejection, protocol
enforcement, and release-lineage hashes. A target-free remote construction smoke
must additionally verify DLO2 tensor shapes, stiffness values, rest-length
shapes, and source provenance before the DLO1 gate may authorize fresh DLO2.

That construction smoke passed on `gpuserver4090` GPU 1. It also established
exact DLO1 rest-geometry and rest-length parity with the original constructor.
The checksummed evidence is stored in
`results/sota/deform_dlo2_initialization_amendment_v1/construction_smoke.json`.
