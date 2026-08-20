# JAX-FEM source qualification result v1

## Decision

The exact pinned JAX-FEM runtime passed the two-action source-physics gate and
failed the separately frozen source-value arm at its outcome-blind full-horizon
physical gate. JAX-FEM is therefore recorded as `source-physics-qualified`, but
this small-strain quasistatic arm is not source-value-qualified, recommended,
or allowed to replace an incumbent backend.

The prefix and future object outcomes were not opened, so this experiment has
no point-accuracy, energy-score, calibration, or state-of-the-art result. Both
selected outputs are byte-identical copies of the registered incumbent
predictions. No DEFORM, target, confirmation, Causal4D, or held-v8 artifact was
read or modified.

## Frozen runtime and custody

The native runtime is JAX-FEM revision
`82c6993c16704e38611f9cb91a5b70f1c690daee`, package version 0.0.12,
with runtime ID
`20c46dfa402712247416730e82289d4d4cd46096cab8c15b49ddb84a69d02a81`.
The source-physics implementation was frozen at BayesianPhysTwin commit
`b09c44c5f0f82ff6a1aa4942c64ff64ba5d82f98`. The source-value model and
protocol were frozen before outcome access at commit
`5eb25d4c462d599431e4d0c0ee10a6f36e4805bb`; the no-outcome physical-gate
finalizer was frozen at `03ef71dd826df2151302bb2f91c506a6422dcde4`.

| Artifact | SHA-256 |
| --- | --- |
| Source-physics result | `ec8c7cb9b9e1a7f833d7857fc51ae3f86d83175bad9336d423d6d8856cacfbcf` |
| Backend qualification | `68140e971e6758e5f1be015a0f0606d3dbfea8f97dd1541f5cea972659d9361c` |
| Sealed full-horizon grid | `3bb6bf8afa878e7fd262344d8cb4ec3260fd16303e2776d8f884cc0d9d675414` |
| Pre-prefix rejection | `39cd7fdda39673f8fb102e452d19e1f37ac0b6d786fbd16c5a5d66e52610a019` |

The full grid contains 768 deterministic quasistatic solves: 58 frames for
`double_lift_zebra` and 198 for `double_stretch_zebra`, each evaluated at
three Poisson ratios. Predictions were generated from frame-zero geometry and
the known action only. The compact grid records `prefix_outcomes_read=false`,
`future_outcomes_read=false`, and
`target_or_held_out_artifact_read=false`.

## Source-physics result

The ten-frame source-physics gate passed deterministic replay, zero-action
drift, rigid-coordinate equivariance, mesh sensitivity, bounded Poisson
sensitivity, Young's-modulus invariance, source-node identity, physical sanity,
and exact incumbent fallback.

| Source group | Action response | Mesh sensitivity | Poisson sensitivity | Determinant range |
| --- | ---: | ---: | ---: | ---: |
| `double_lift_zebra` | 8.588 mm | 0.754% | 0.153 mm | 0.961 to 1.065 |
| `double_stretch_zebra` | 0.901 mm | 2.786% | 0.020 mm | 0.992 to 1.012 |

This result qualifies the exact native integration and the local physical
model over the registered short interval. It does not imply full-horizon
stability.

## Full-horizon rejection

The full-horizon gate required contact-projection error at most 20 mm, maximum
node displacement at most 350 mm, and every tetrahedral deformation determinant
within `[0.5, 2.0]`.

| Group | Poisson ratio | Contact projection | Min determinant | Max determinant | Max displacement | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `double_lift_zebra` | 0.20 | 14.573 mm | 0.664 | 1.402 | 291.302 mm | pass |
| `double_lift_zebra` | 0.35 | 14.573 mm | 0.687 | 1.571 | 291.848 mm | pass |
| `double_lift_zebra` | 0.45 | 14.573 mm | 0.471 | 1.803 | 290.544 mm | fail determinant |
| `double_stretch_zebra` | 0.20 | <0.001 mm | -8.838 | 37.194 | 567.136 mm | fail |
| `double_stretch_zebra` | 0.35 | <0.001 mm | -8.163 | 37.517 | 570.165 mm | fail |
| `double_stretch_zebra` | 0.45 | <0.001 mm | -19.149 | 34.374 | 568.510 mm | fail |

The lift case is close to admissible, but its high-Poisson member crosses the
registered inversion threshold. All stretch members contain severe element
inversions and exceed the displacement limit. The final ensemble spreads are
only 0.337 mm and 0.386 mm, so the admitted Poisson uncertainty does not cover
this structural failure.

## Consequence

This is a rejection of the frozen linear small-strain, independent-frame
quasistatic formulation under the registered full actions, not a rejection of
JAX-FEM as a solver or backend interface. A future JAX-FEM arm would need a
genuinely new large-deformation formulation with non-inversion controls and a
newly frozen source protocol. It must not loosen these gates or tune on the two
opened actions.

The retained fallback hashes are
`e40e24f33dffd48883f969c38d5fba2410a34cc12f9daf01d3910cb75f69890b`
for lift and
`c0d169b4df7077aab6e81a7338cb86db5c5f5772e5011f42804c75a7c98dbefd`
for stretch. DEFORM remains the protected strong result and shares only the
portable rollout contract with this experiment.
