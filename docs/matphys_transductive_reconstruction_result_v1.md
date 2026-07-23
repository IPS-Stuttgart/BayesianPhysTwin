# MatPhys all-frame reconstruction result

## Status

This is the completed result of the frozen
`matphys-offline-all-frame-reconstruction-v1` control. One model was fit per
case for 200 epochs with the released test interval included in the objective.
All 22 cases and all failures are retained.

This is **offline reconstruction**, not causal future prediction. Future object
observations and released test outcomes were used during fitting.

## Full-cohort result

Lower is better. The case-balanced mean is primary.

| Result | CD (mm) | Manual-track error (mm) |
|---|---:|---:|
| Published rounded MatPhys reference | **8.000** | **15.000** |
| Runnable public-artifact path, case-balanced | 12.212 | 26.881 |
| Runnable public-artifact path, frame-weighted | 13.117 | 30.875 |

The case-balanced result is 52.65% worse in CD and 79.20% worse in track
error than the published rounded point. The local public-artifact path
therefore fails to reproduce the published aggregate despite using the easier
all-frame information regime.

Eight of 22 cases are descriptively below both rounded aggregate reference
values. That count is not a per-case statistical gate; it only shows that the
failure is heterogeneous rather than universal. The largest identity errors
are:

| Case | CD (mm) | Track error (mm) |
|---|---:|---:|
| `single_lift_cloth_4` | 29.692 | 105.660 |
| `single_lift_cloth` | 29.515 | 79.016 |
| `single_lift_cloth_3` | 19.065 | 68.193 |
| `single_push_sloth` | 27.701 | 46.558 |
| `weird_package` | 12.586 | 39.388 |

The preregistered two-case continuation gate passed before the other 20 cases
were run. The complete cohort shows why that gate was only a compute-spending
check and not evidence of benchmark-wide transfer.

## Public-artifact boundary

The MatPhys repository does not release the generated `node_sem.npz` and
`train_ready.pt` products used by the paper pipeline. This control therefore
uses the previously frozen `global-onehot-single-part-v1` proxy: the released
object material class is supplied, every node shares one part, and the simple
decoder's unused semantic tensor is zero. Every proxy and input is hash-bound
in the per-case training audit.

Consequently, this result does not establish that the paper's private artifact
pipeline cannot attain 8/15 mm. It establishes the narrower and reproducible
fact that the pinned public implementation plus the disclosed available-data
proxy does not reproduce that point. The published MatPhys row must remain an
external paper value rather than a locally reproduced SOTA baseline.

## Decision

Do not use this all-frame family as a causal Bayesian-PhysTwin backbone and do
not tune its opened 22-case outcomes. Together with the failed causal-prefix
development gate, this result closes the current MatPhys spring-only path.
Further work should target guarded Bayesian state/discrepancy updates rather
than additional spring-field fitting.

The all-frame control is retained as a reproducibility result and as evidence
that future access alone does not resolve the public-artifact gap.

## Provenance

- MatPhys source: `c16b858dfb79bf21024ead24b45a710600de7b4f`
- Bayesian-PhysTwin evidence code: `d938d9e`
- Run root: `gpuserver4090:/home/florianpfaff/matphys-transductive-bpt-v1/runs/matphys-fitall-22-v1`
- Cases / official test frames: 22 / 735
- Aggregate: `results/sota/matphys_transductive_reconstruction_v1/aggregate.json`
- Aggregate SHA-256: `949e44db000f36a3d467aa26d1b49a1679a5753a141c5931aa947cd403282cc1`
- Two-case gate SHA-256: `c04a8f0cef017cf3b21e4feb5930f0421041d4dbb4360852211251f04261d62c`
- Verification: 9 focused tests passed; 1,038 native POSIX tests passed
  and 5 skipped; changed Python files pass Ruff

The aggregate was emitted only after verifying every result contract and the
recorded size and SHA-256 of all 22 best checkpoints, trajectories, and
training audits.
