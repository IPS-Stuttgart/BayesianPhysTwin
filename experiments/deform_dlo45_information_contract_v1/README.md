# DEFORM DLO4/DLO5 information-contract adapter v1

This experiment replays the previously completed, source-frozen DEFORM DLO4/DLO5
decision-identifiability study through the Prob4D probabilistic 4-D
information-contract benchmark.

It is an **exact retrospective adapter**, not a new target experiment. The
adapter does not refit a model, select another policy, tune on held trajectories,
open a new cohort, or replace any failed case. It reconstructs the per-decision
finite support from the exact retained source model and the pinned public DEFORM
evaluation trajectories, then verifies that the benchmark reproduces every
published DLO-level and trajectory-level RMSE and every action count.

## Bound inputs

- Original source-only workflow run: `33473378340`
- Original source artifact: `9787311310`
- Original source head: `38b2ea56471923e63b64cfe24bf3f691aad5d5e0`
- Source model SHA-256:
  `a43aed43cd563ee47358e48cab84829dc7eebc77d97725721a11b228f3b6b7f0`
- Public DEFORM repository revision:
  `b73b8b8ecc033caefa693fab7898741d4e6dbeff`
- Prob4D benchmark revision:
  `f25b0cdb0a258e1b2ef276d25a723c2cf3a9fb4f`

The exact identities and the prohibition on target tuning, retries, new target
access, and payload redistribution are frozen in
`request.json`.

## Benchmark suites

The adapter emits six suites over the same 532 decisions and 28 complete
trajectory groups:

| Suite | Tasks | Role |
| --- | --- | --- |
| `fallback` | forecast | caller-owned physical fallback |
| `certificate` | forecast, decision | exact quotient-regret policy and realized held regret |
| `jeffrey_point` | forecast | one canonical complete belief |
| `kernel_point` | forecast | kernel-weighted point belief |
| `map_point` | forecast | nearest/MAP complete state |
| `oracle` | forecast | held-out diagnostic lower bound |

For the certificate suite, each case contains the exact source-supported
hypotheses, quotient labels and masses, finite action-loss matrix, reported
worst-case regret, selected action, fallback action, and held realized losses.
The benchmark independently recomputes the certificate and verifies its
admission and fallback policy.

One benchmark case is one decision window. `group_id` is the complete DLO
trajectory, so aggregate equal-group reporting does not treat the 19 correlated
windows in one trajectory as independent experimental units.

## Information order

For every window the adapter:

1. constructs the observation from the five-frame prefix and registered future
   endpoint action;
2. reconstructs the exact source-neighbor support, quotient posterior, Jeffrey
   correction, and all method actions;
3. verifies parity with the frozen implementation;
4. only then slices the held internal-node trajectory and computes realized
   action losses; and
5. seals a SHA-256-bound, pickle-free benchmark payload.

The public pickle carrier co-locates the permitted observation and held outcome;
this adapter preserves the original semantic slicing but cannot create byte-level
channel separation retroactively.

## Reproduction

The permanent workflow retrieves the exact historical source artifact, checks out
only the pinned public DLO4/DLO5 evaluation paths, installs the exact Prob4D
benchmark revision, exports all six suites, evaluates them, and runs the strict
reference validator.

A local equivalent is:

```bash
python -m experiments.deform_dlo45_information_contract_v1.export export \
  --request experiments/deform_dlo45_information_contract_v1/request.json \
  --protocol experiments/deform_dlo45_decision_identifiability_v1/protocol.json \
  --dataset-root /path/to/DEFORM/data_set \
  --model /path/to/source_model.npz \
  --source-result /path/to/source_result.json \
  --source-seal /path/to/source_seal.json \
  --reference-result \
    results/science/deform_dlo45_decision_identifiability_v1/\
official-dlo45-one-shot-20260901-v1/target_result.json \
  --output-root /tmp/dlo45-information-contract

for method in fallback certificate jeffrey_point kernel_point map_point oracle; do
  python -m prob4d.information_contract_benchmark evaluate \
    "/tmp/dlo45-information-contract/${method}/suite.json" \
    "/tmp/dlo45-information-contract/${method}/benchmark_result.json"
done

python -m experiments.deform_dlo45_information_contract_v1.export validate \
  --output-root /tmp/dlo45-information-contract \
  --reference-result \
    results/science/deform_dlo45_decision_identifiability_v1/\
official-dlo45-one-shot-20260901-v1/target_result.json \
  --destination /tmp/dlo45-information-contract/validation.json
```

## Retained evidence boundary

The workflow retains suite manifests, benchmark result files, dataset and source
identities, and exact replay validation. Per-case NPZ carriers are transient and
are not redistributed. They can be regenerated from the pinned public dataset
and retained source artifact while that artifact remains available.

A positive replay establishes that one public real-data experiment can be
expressed and independently checked through the common benchmark contract. It
does not create an independent replication, validate the registered support on
new objects, establish target-domain regret coverage, calibrate a probabilistic
provider, or authorize robot deployment.
