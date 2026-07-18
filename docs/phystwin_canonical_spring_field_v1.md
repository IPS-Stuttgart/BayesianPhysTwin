# Canonical spring-field transfer test

Run date: 2026-07-18

Status: implementation validated; locked source-transfer gate failed; no future
metric was opened.

## Hypothesis

NeuSpring reports that a canonical coordinate-conditioned spring field and
piecewise topology account for most of its gain over PhysTwin. Earlier
Bayesian-PhysTwin controls tested hard spatial regions and zero-order candidate
banks, but not a differentiable smooth field centered on every released
PhysTwin spring.

The new `canonical_basis` parameterization is

```text
log(k_e) = log(k_e^PhysTwin) + sum_j w_ej beta_j,
```

where `w_ej` are normalized Gaussian-RBF weights over canonical object-spring
midpoints. Sixteen centers are selected deterministically by farthest-point
sampling. Controller springs use one separate coefficient. Equal object
coefficients recover a global scale, while all-zero coefficients reproduce the
released spring field exactly.

This is a compact, auditable proxy for spatial spring associativity. It is not
a reproduction of NeuSpring's learned tri-plane field.

## Implementation checks

The official Warp backend expands the basis coefficients inside the captured
simulation graph before every step, so gradients propagate through the spring
field. The runner writes the complete basis, center indices, length scale,
weights hash, coefficients, and induced log-scale range.

On `single_lift_sloth`, the zero-coefficient trajectory was bit-identical to
the same-run exact teacher. The historical released trajectory differed by
`0.0316 mm` mean node-frame distance, within the known Warp replay floor. The
identity summary is archived at
`results/sota/diagnostics/phystwin_canonical_spring_field_v1/identity_summary.json`.

Six focused basis tests cover identity centering, controller separation,
rigid-transform invariance, deterministic center selection, rank clipping, and
invalid hyperparameters.

## V1 scale failure

The first protocol was frozen at
`configs/sota/phystwin_canonical_spring_field_source_v1.json`. Its Adam weight
decay was `0.01`, while the mean Warp likelihood was about `3e-6`. This scale
mismatch collapsed every coefficient to approximately `1e-5`. The best
development epoch improved validation CD by only `0.0065%` and track error by
`0.0255%`, far below the locked `1%` competence gate.

A fit-only diagnostic, without validation access, removed weight decay. The
coefficients then reached meaningful `0.04--0.29` object log scales and fit loss
fell monotonically. V1 remains an immutable failed result; it was not amended.

## V2 development result

V2 was registered before applying the field to a different development
interaction, `double_lift_sloth`. It uses the same rank, spatial scale,
optimizer, and future-blind split, but relies on the low-rank RBF field itself
for regularization and enforces a factor-two post-fit plausibility bound.

| Metric | Exact teacher | Canonical field | Improvement |
| --- | ---: | ---: | ---: |
| Validation CD | 9.343 mm | 8.633 mm | 7.60% |
| Validation track | 18.229 mm | 17.090 mm | 6.25% |
| Balanced | - | - | 6.92% |

The object field ranged from `-0.429` to `-0.003` log scale and the controller
coefficient was `+0.067`, all within the factor-two bound. V2 therefore passed
the development competence gate.

## Frozen source result

The four transfer objects were then run with the unchanged V2 configuration.
Each fit saw only the first 75% of its released observation prefix; checkpoint
selection used the remaining prefix suffix. No future object observation,
future control, or future metric was available.

| Source case | CD improvement | Track improvement | Selected field | Plausible |
| --- | ---: | ---: | --- | --- |
| `single_lift_zebra` | 0.00% | 0.00% | teacher | yes |
| `single_lift_cloth` | 20.42% | 7.79% | epoch 11 | **no** |
| `single_lift_rope` | 0.00% | 0.00% | teacher | yes |
| `single_lift_dinosor` | 0.00% | 0.00% | teacher | yes |

Cloth required object log scales from `-1.443` to `+1.690` and a controller
scale of `+0.812`: up to `5.42x` stiffness and outside the preregistered
factor-two range. The other three objects selected the exact teacher. Thus the
plausible non-teacher and two-metric-win counts are both `0/4`, below the
required `3/4`. The machine-readable gate result is
`results/sota/diagnostics/phystwin_canonical_spring_field_v2/source_gate_summary.json`
with SHA-256
`7391b8f36557c10411b19bcb292bfc16c56a3ae7d3fe59c097f3b3eedf8fe1c4`.

## Conclusion

The experiment rejects another incremental static-refit route. A smooth field
can materially help a particular interaction, but it does not transfer as a
credible per-object family. This agrees with the earlier zero-order and
part-pair failures and does not authorize a 19-case future run.

Closing the roughly 20% gap to MatPhys now requires information absent from a
single interaction prefix: a shared material/part prior learned across objects,
or an action-dependent constitutive/contact model trained across interactions.
Another static spring bank or output residual is not justified by the evidence.
