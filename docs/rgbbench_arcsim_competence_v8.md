# RGBench ARCSim competence v8

## Purpose

This is a target-free numerical qualification of ARCSim as a deterministic
thin-sheet backbone for the RGBench cloth source case. It does not compare a
prediction against an RGBench point cloud and cannot support an accuracy or
state-of-the-art claim.

The prior Codim-IPC arm was deterministic and exact over 0.1 s, but its
full-horizon replay stalled after 48 of 1636 steps. Classic IPC was also audited
and rejected as a direct adapter because its three-dimensional elastic state
requires tetrahedra while RGBench exposes identity-bearing cloth surfaces.
ARCSim is tested because it natively evolves triangular thin sheets, supports
fixed topology, and provides time-dependent node handles.

## Compatibility boundary

The official ARCSim 0.2.1 release no longer builds against the current system
toolchain. The bound patch
`third_party/patches/arcsim_rgbbench_compat_v8.patch` makes only build and
serialization compatibility changes:

1. user-selectable Eigen and Boost include prefixes;
2. direct compilation of the bundled JSON library;
3. an Eigen sparse-solver replacement for the obsolete TAUCS wrapper;
4. a typo fix in an unused sparse debug serializer;
5. 17-digit OBJ vertex serialization.

It does not alter cloth energies, constraints, collision mechanics, or the time
integrator. The protocol binds the official release archive, patch, modified
sources, static libraries, and executable by SHA-256.

## Frozen physical interface

The source garment is `green_tshirt/fling/01`. Its 9,865 source vertices and
19,555 faces remain in their released order. Remeshing, collision, proximity,
separation, strain limiting, plasticity, and the pop filter are disabled in
this mechanics-only gate. Two known RGBench pin identities receive the released
actuator trajectories through ARCSim node handles.

RGBench provides metric volume density, mass, surface area, Young's modulus,
and Poisson ratio. The adapter derives shell thickness from mass, area, and
volume density, then converts the isotropic plane-stress membrane and thin-shell
bending coefficients into ARCSim's tabulated material format. No observed
future garment state enters that conversion.

## Frozen gate

Two isolated, single-thread, 0.1 s replays must both satisfy:

- byte-identical finite final arrays;
- exactly 9,865 output vertices and 10 simulation steps;
- maximum moving-handle error at most 0.01 mm;
- mean cloth displacement at least 0.01 mm;
- wall time at most 600 s per replay.

The motion check prevents an exact no-op from passing. The relaxed handle
threshold acknowledges that ARCSim uses penalty handles rather than projected
Dirichlet constraints, while remaining negligible against RGBench errors.

## Information boundary

Allowed inputs are the released source mesh, material metadata, source actuator
trajectories, and frozen pin identities. Allowed diagnostics are numerical
replay equality, topology preservation, pin error, displacement magnitude, and
wall time.

All segmented point-cloud names and coordinates, and every source, calibration,
or target accuracy outcome, remain forbidden. Passing this gate authorizes only
a separately frozen full-horizon target-free qualification. Failure closes the
backend without opening an RGBench point-cloud outcome.
