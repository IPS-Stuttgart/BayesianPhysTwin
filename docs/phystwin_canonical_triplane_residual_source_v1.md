# Canonical triplane residual source gate

Run date: 2026-07-19

Status: source gate failed; no target artifact was opened.

## Hypothesis

The previous shared linear, graph-local MLP, and PGRD residual families failed
to transfer across interactions. Published results from NeuSpring and
DeformMaster suggest two useful inductive biases that those models omitted:
canonical material coordinates and spatially local grid processing.

This experiment therefore learned residual dynamics with three shared
canonical feature planes. A deterministic right-handed PCA frame maps each
released PhysTwin graph into material coordinates. Point features are
bilinearly splatted to XY, XZ, and YZ planes, processed by a shared convolution
tower, and sampled back to predict the next residual velocity. A 16-dimensional
case latent is the only quantity adapted on a held-out interaction.

This is a source-transfer test of a residual dynamics family. It is not a
reproduction of NeuSpring's per-case spring field or DeformMaster's
particle-grid simulator.

## Information boundary

The 17 registered source interactions were divided into three whole-case
folds. Complete outcomes supervised only the non-held-out interactions. For
each held-out interaction:

1. global network weights remained frozen;
2. only the case latent was fitted on frames `[0, fit_end)`;
3. velocity and residual caps used only that prefix;
4. observations after `fit_end` were held at the final visible prefix value;
5. scoring used `[fit_end, train_end)`;
6. no target file or trajectory was read.

An initial implementation audit found that temporal filling could back-fill a
prefix point from later visibility. That run was terminated before completion,
its processes were killed, and no result was retained. The clean run used a
causal fill boundary and passed a future-mutation invariance test before any
gate result was accepted.

## Frozen result

Three seeds (`17`, `43`, and `101`) were trained per fold. Predictions were
ensembled and compared with exact endpoint residual persistence at four frozen
blend weights.

| Dynamic blend | CD ratio | Track ratio | Balanced improvement | Both-win folds | Worst case ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 1.00521 | 1.00071 | -0.296% | 0/3 | 1.04392 |
| 0.50 | 1.01342 | 1.00278 | -0.810% | 0/3 | 1.10168 |
| 0.75 | 1.02340 | 1.00630 | -1.485% | 0/3 | 1.16400 |
| 1.00 | 1.03605 | 1.01204 | -2.405% | 0/3 | 1.22659 |

The mechanically selected `0.25` blend worsened equal-case CD by `0.521%`
and track error by `0.071%`. Fold CD/track ratios were:

| Fold | CD ratio | Track ratio |
| --- | ---: | ---: |
| 0 | 1.01213 | 0.99862 |
| 1 | 1.00166 | 1.00282 |
| 2 | 1.00116 | 1.00067 |

Only `single_lift_dinosor` showed a material two-metric benefit
(`-1.99%` CD and `-1.36%` track). Increasing reliance on the learned dynamics
made the aggregate and worst-case results monotonically worse. The locked gate
required at least `3%` balanced improvement, both aggregate metrics to improve,
at least two both-win folds, and no case ratio above `1.05`; it therefore
failed unambiguously.

The accepted summary SHA-256 is
`4a98010ee2bf2cacbf581f6e4360a6ced204318e7ef33c7e7e2bca9b63cfb15b`.
Machine-readable evidence is archived under
`results/sota/diagnostics/phystwin_canonical_triplane_residual_source_v1/`.

## Interpretation

Canonicalizing the object state and processing residuals through shared local
feature planes does not make residual evolution transferable on these
interactions. Endpoint persistence remains stronger. The monotonic blend
failure also argues against treating the result as an underweighted useful
signal.

This closes the pooled canonical-triplane residual branch. It does not reject
three distinct alternatives:

1. a per-case canonical spring field optimized inside the official simulator;
2. an equivariant state model with canonicalized actions rather than a
   canonicalized state;
3. a state- and regime-dependent generalized-force correction propagated
   through the simulator.

The highest-value successor combines the latter two: a gravity-aware
equivariant graph network should predict bounded generalized forces inside the
pinned upstream Warp runtime. Relative positions, relative velocities, spring
strain, and controller-relative actions provide the equivariant inputs; a
small prefix-only latent permits interaction adaptation; zero predicted force
must exactly reproduce the released simulator. This is materially different
from both the rejected pooled GNN readout and the rejected constant-force
audit.
