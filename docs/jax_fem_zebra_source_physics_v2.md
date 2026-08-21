# JAX-FEM finite-deformation source qualification v2

**Decision:** passed for the exact pinned runtime. This authorizes only the
separately frozen source-value stage; it is not an accuracy or backend-selection
result.

## Question

Does replacing the terminally failed small-strain JAX-FEM v1 arm with one
source-independent, finite-deformation constitutive amendment produce a
numerically admissible backend on the same two already-open source actions?

This gate cannot establish predictive value. It may authorize a separately
frozen source-value comparison, but only if every numerical, provenance,
fallback, and information-order check passes.

## Causal isolation

The v2 protocol is an overlay on
`configs/sota/jax_fem_zebra_source_physics_v1.json` at SHA-256
`cc8b0fcdfb32de925bc2cafee53b2600385f150eba6e5f514f42e097d9623047`.
It therefore inherits the exact `double_lift_zebra` and
`double_stretch_zebra` source roster, source-input hashes, incumbent bytes,
mesh policy, and rigid-contact construction. The engine revision and all
runtime packages also remain fixed. The only scientific amendment is the
large-deformation formulation and its deterministic continuation solver.

The v1 physical-gate failure remains part of the evidence. V2 does not replace
or reinterpret it.

## Frozen formulation

The energy density is the stable Neo-Hookean model of Smith, de Goes, and Kim:

```text
psi(F) = mu / 2 * (tr(F^T F) - 3) + lambda / 2 * (det(F) - alpha)^2
alpha  = 1 + mu / lambda
```

JAX automatic differentiation supplies the first Piola stress and tangent.
Each source-frame interval is solved by the pinned JAX-FEM Newton path with a
SciPy sparse direct solve, line search, and the previous displacement as the
initial guess. The base arm takes one continuation step per registered source
frame. The refinement arm inserts exactly one midpoint transform, with each
contact-patch rotation projected deterministically back onto `SO(3)`.

The runtime identity is
`0c1a24a70c805eb6ade62d176fafd574b3fc1c07fef8ca80592db6bb9ad23d15`.
The protocol SHA-256 is
`eafcb9ff2a9c2b6b94c438424b0480e6f3b3008f147c99a1a5120e34b5a94b0c`.

## Gates

The first ten source-input frames are used without object observations. The
runner checks deterministic replay, rest equilibrium, rigid-coordinate
equivariance, one-step versus two-step continuation, 25/30 mm connectivity
sensitivity, Poisson sensitivity, Young-modulus invariance under pure
Dirichlet loading, stress objectivity, contact projection, mesh identity, and
deformation determinants. The frozen scientific determinant interval is
`[0.35, 3.0]`; a separate `0.05` hard floor stops an invalid inverted solve.

The incumbent prediction is read only for frame-zero query parity and copied
byte-for-byte as the fallback. No source outcome, target, reserve, or held-v8
artifact is permitted.

## Registered command

After deploying one clean committed archive into the pinned Python 3.12
runtime, the source custodian runs exactly one command of this form:

```bash
python scripts/remote/run_jax_fem_hyperelastic_source_qualification_v2.py \
  --protocol configs/sota/jax_fem_zebra_source_physics_v2.json \
  --group-root double_lift_zebra=/absolute/source/double_lift_zebra \
  --group-root double_stretch_zebra=/absolute/source/double_stretch_zebra \
  --output-dir /new/nonexistent/source-only/output \
  --repo-root /exact/clean/source
```

An existing output directory is a hard error. Failure is preserved without
retry, parameter changes, group replacement, or outcome access. A pass permits
only the pre-registered full-horizon source prediction and its physical gate.

## Frozen result

The one source-only execution used BayesianPhysTwin commit
`54178900fc566d8493fabc6b6808c5d3908b539c` and exact source archive SHA-256
`527e9bb9f38a05c9150354d66555ca9e7af2ef0232fec8b8b2c325dceed8b1bf`.
It completed 218 native nonlinear solves without replacement or retry.

| Artifact | SHA-256 or content ID |
| --- | --- |
| Source-physics result | `10c2bd94436b3b4414f30becd859667ddab88c0445aa17c950186fc6e1f434e3` |
| Backend qualification | `e2f0797d0778b6143a076debb4b2596baffd430477e55b0499b45d1b68d51ef6` |
| Qualification artifact ID | `820df616afcd911af2999aa3b208f8d2da1e2acbe62521bc9d1980fc317aba50` |
| Result ID | `3f59667602963b0623d1c0c0df687ca90c009415f9d47ec1f890a53f6764da45` |
| Lift trajectory archive | `fba703f07c1b8ff2190f397aa3cc3c820ba7303a0b3bc426484684466ff240ea` |
| Stretch trajectory archive | `f4ce6ee982452f77b90e38f029a95c59c23e3d6819285a6b48b058ede1cd0c48` |

| Source group | Response | Mesh sensitivity | Continuation error | Poisson sensitivity | Determinants |
| --- | ---: | ---: | ---: | ---: | ---: |
| `double_lift_zebra` | 8.442 mm | 0.748% | `1.15e-9` | 0.111 mm | 0.959 to 1.067 |
| `double_stretch_zebra` | 0.900 mm | 2.744% | `4.89e-7` | 0.0128 mm | 0.990 to 1.013 |

Both groups had exact replay, zero rest drift, sub-femtometre rigid-coordinate
error, source-node parity, bounded contact projection, Young-modulus invariance,
and byte-identical fallback. The result and qualification replay with
`source_object_outcomes_read=false` and
`target_or_held_out_artifact_read=false`.
