# JAX-FEM source-physics qualification v1

## Question

Can the exact pinned JAX-FEM runtime construct and solve a physically admissible
finite-element model of two already-open source actions before any source object
outcome is read?

This is a numerical and custody gate, not an accuracy result. Passing authorizes
only a separately frozen source-value comparison. It does not modify the
specialized DEFORM implementation, its evidence, or its exact fallback.

## Source boundary

The gate uses `double_lift_zebra` and `double_stretch_zebra`. It may read only:

- frame-zero material nodes;
- the known full controller trajectory;
- registered controller-to-material attachment weights; and
- the existing incumbent prediction for frame-zero parity and byte-exact
  fallback.

Prefix and future object observations remain closed. Target, confirmation, and
held-v8 artifacts are outside the protocol.

## Frozen model

The source points are tetrahedralized deterministically with SciPy/Qhull options
`Qbb Qc Qz Q12`. Tetrahedra longer than 25 mm or below the frozen volume-to-edge
shape ratio are removed, then oriented positively. A 30 mm edge limit is the
registered connectivity-sensitivity control. Both meshes must retain every
source identity in one connected component and reproduce their locked cell
counts.

The registered controller map provides a target for each attached material
node. Those targets are not imposed independently. Nodes connected within
15 mm form contact patches, and each patch is projected to its least-squares
rigid `SE(3)` motion at each source frame. This models a coherent grasp surface
and prevents the controller-marker field from being mistaken for local material
strain. The projection error is reported and capped at 5 mm during the ten-frame
qualification interval.

The constitutive law is small-strain isotropic linear elasticity on `TET4`
elements. Each source frame is an independent quasistatic solve through the
pinned JAX-FEM `scipy-spsolve` path. This is intentionally the simplest stable
model family that survived target-free mesh and contact probes; the rejected
neo-Hookean probe converged but inverted cells and is not part of this gate.

## Identifiability boundary

Pure displacement-controlled quasistatic loading does not identify the absolute
Young's modulus: scaling every stiffness by the same factor leaves the
displacement solution unchanged. The gate verifies that invariance instead of
claiming a Young's-modulus posterior. Poisson ratio does change the displacement
field and is the only material ensemble axis admitted to a later source-value
test (`0.20`, `0.35`, `0.45`). No gradient or differentiability claim is made.

## Probes and gates

For source frames `[0, 3, 6, 9]`, the runner executes a base solve twice, a
zero-action solve, a fixed rigid-coordinate transform, the 30 mm connectivity
control, low/high Poisson-ratio arms, and low/high Young's-modulus terminal
controls. It also solves every frame `[0, ..., 9]` and compares the shared
source frames. For this path-independent quasistatic model, that field records
shared-load-step parity rather than a dynamic time-step convergence claim.

Qualification requires:

1. exact repeat arrays;
2. at most `1e-10 m` zero-action drift;
3. at most `1e-6 m` rigid-coordinate equivariance error;
4. shared-load-step relative error at most `1e-6`;
5. 25/30 mm mesh sensitivity at most 5% of the action response;
6. a measurable but bounded Poisson-ratio response;
7. Young's-modulus displacement invariance within `1e-9 m`;
8. finite, non-inverted deformation with determinants in `[0.5, 2.0]`;
9. exact source-node identity and frame-zero parity; and
10. byte-identical incumbent fallback.

Any failed gate keeps source object outcomes closed. A pass only advances this
exact JAX-FEM runtime to a new source-value protocol; it is not evidence that
JAX-FEM improves prediction, uncertainty, Causal4D, or state of the art.
