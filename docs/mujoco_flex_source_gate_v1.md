# MuJoCo volumetric Flex source gate v1

**Decision:** rejected at the source-physical gate. No source outcome or target
artifact was opened, and no retry or parameter search is authorized.

## Question

Can the exact pinned MuJoCo 3.9 volumetric Flex runtime replay the registered
irregular tetrahedral PhysTwin source geometry under moving rigid-patch
attachments while retaining valid element orientation?

The source-only audit covered three attachment architectures: direct material
vertex ownership by mocap bodies, a reduced regular trilinear grid, and dynamic
material vertices connected to independent mocap targets by native `connect`
constraints. None cleared every predeclared numerical gate.

## Final architecture

The final source-independent amendment kept every original tetrahedral vertex
dynamic, attached each registered contact vertex to a separate rigid target
with a native point constraint, and applied the minimum conservative blend of
lumped nodal masses required to cap the mass condition number at `100`. The
blend preserves total reference mass and changes neither source geometry nor
action.

Its synthetic smoke passed 20 native steps with minimum determinant
`0.9987218905`. The sole frozen `double_lift_zebra` replay then failed at
substep `3/334`: the minimum element determinant was `-0.5294351619`, below the
hard `0.35` floor. This rules out source-value scoring and target evaluation
for this architecture.

## Decision boundary

The compact receipt is
[`results/sota/diagnostics/mujoco_flex_source_gate_v1/failure.json`](../results/sota/diagnostics/mujoco_flex_source_gate_v1/failure.json).
It binds the exact engine wheel, adapter bytes, source-input hash, derived mesh,
frozen physical parameters, and terminal failure. The direct and reduced-grid
architectures remain available for reproducibility, but neither is an admitted
paper backend. The incumbent remains the byte-exact fallback.
