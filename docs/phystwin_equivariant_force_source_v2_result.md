# Equivariant generalized-force v2 pretraining result

Date: 2026-07-24

Status: source target QA passed; Stage 1 has not started.

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

Before Stage 1, the previously underspecified Stage-2 replay and seed
aggregation choices were independently locked in
`configs/sota/phystwin_equivariant_force_stage2_v1.json`. This amendment does
not change this v2 source archive or its QA evidence.
