# Equivariant generalized-force v2 pretraining result

Date: 2026-07-24

Status: source target QA passed; the subsequently frozen Stage 1 failed.

## Immutable inputs

- implementation commit: `ffb87eab`
- exact source archive SHA-256:
  `fdb13f40443bedeadb05e5cff3d4f478530f89e0ffa521b7c5945c5c124378c1`
- protocol SHA-256:
  `1178ffe1545158225818723c700991f76d730c3627ab09644b73f2a14f53a171`
- episode-build summary SHA-256:
  `68d0e89460439a96099787ce7c8d218f82c8c7a06f9c8722fa8511a4851e7d02`
- native environment: Python 3.10.12, NumPy 1.26.4, SciPy 1.13.1,
  PyTorch 2.4.0+cu121

The generated summary is stored at
`results/sota/phystwin_equivariant_force_source_v2/episode_build_summary.json`.
Large typed episode arrays remain on `gpuserver4090` under the exact archived
source tree.

## Pretraining gate

All 17 released source episodes were built and checksum-validated. No target
artifact was opened.

| Check | V1 preflight | V2 |
| --- | ---: | ---: |
| Unit contract | incorrectly labeled Newtons | native Warp simulator force |
| Per-case cap | fixed `0.5` | prefix-only robust scale |
| Prefix cap fraction | 43.43% to 91.51% | 1.66% to 4.96% |
| Scale range | not meaningful | 2.21 to 10.65 simulator units |
| Source target QA | failed | passed 17/17 |
| Stage 1 run | no | no |

V2 therefore passes the frozen maximum 10% prefix cap-fraction gate. Every
released graph also satisfies the declared unit-mass simulator contract.

This is not a prediction result. It establishes only that the inverse-dynamics
targets are numerically usable, causally prefix-scaled, and honestly labeled.
The held-out source suffix statistics were not inspected.

## Verification

- focused native suite: 35 passed
- native CPU full suite with GPUs hidden: 1,055 passed, 4 skipped
- source-build validator: `17 True False`
  (`episode_count`, `target_QA_passed`, `target_artifacts_opened`)

## Next gate

Stage 1 may start only after a configured GPU is explicitly released by the
independent registered experiments. It will run the frozen three-fold,
three-seed crossfit and test normalized force-target competence. Passing Stage
1 still does not authorize a simulator or state-of-the-art claim; it only
permits the information-matched official-Warp Stage 2.

Stage 1 was later executed from the exact preflighted deployment and failed
with 0/3 passing folds and 0.97% mean held-out improvement against its 10%
gate. The immutable result is documented in
`docs/phystwin_equivariant_force_stage1_v2_result.md`. Stage 2 was not
authorized and no target artifact was opened.

Before Stage 1, the previously underspecified Stage-2 replay and seed
aggregation choices were independently locked in
`configs/sota/phystwin_equivariant_force_stage2_v1.json`. This amendment does
not change this v2 source archive or its QA evidence.

## Registered Stage-1 execution

Stage 1 may run serially with `source-competence`, or as three independently
recorded folds followed by a mechanical merge. The sharded path changes only
execution scheduling. It preserves the exact registered folds, seeds, source
episodes, model settings, thresholds, and target boundary.

After a GPU is explicitly released, two folds may run concurrently by exposing
one physical GPU to each process while retaining the protocol's `cuda:0`
device string:

```bash
CUDA_VISIBLE_DEVICES=0 bpt-gate-phystwin-equivariant-force \
  source-competence-fold EPISODES PROTOCOL OUTPUT fold_0 --device cuda:0

CUDA_VISIBLE_DEVICES=1 bpt-gate-phystwin-equivariant-force \
  source-competence-fold EPISODES PROTOCOL OUTPUT fold_1 --device cuda:0
```

The remaining `fold_2` runs unchanged on the first released device. A completed
fold record cannot be overwritten. Once all three records exist:

```bash
bpt-gate-phystwin-equivariant-force \
  source-competence-merge EPISODES PROTOCOL OUTPUT --device cuda:0
```

The merge performs no fitting. It requires exact protocol and episode
identities, registered held-out coverage and seed order, prefix-only latent
provenance, expected fold-local paths, and matching model and latent SHA-256
digests. Every fold also binds the same deterministic Stage-1 implementation
hash, and the merged record binds all fold-record hashes. Any incomplete,
relocated, mixed-code, or modified fold blocks the decision.

The CPU-only remote deployment is recorded in
`results/sota/phystwin_equivariant_force_stage1_v2/preflight.json`. It binds
commit `6245d42`, the deployment archive, the Stage-1 implementation modules,
the immutable source episodes, the protocol, the environment, and the exact
fold and merge commands. Its SHA-256 is
`036454b6701ff1eb4583f3666f5cda8683e12ff55fd1caba294580cbf40fd8be`.
This replacement restores the protocol's declared per-node force-cap
semantics; the superseded deployment produced no outcome.
Its status remains
`ready_but_not_started_pending_explicit_gpu_release`.
