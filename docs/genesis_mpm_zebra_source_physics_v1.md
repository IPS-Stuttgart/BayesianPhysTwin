# Genesis MPM source-physics qualification v1

## Question

Can the exact Genesis 1.3.3 MPM runtime pass the numerical and custody checks
required before BayesianPhysTwin is allowed to compare its predictions against
source object outcomes?

This is not a point-accuracy result. It is the frozen source-physics stage of
the backend funnel in `docs/backend_admission_policy_v1.md`.

## Source groups

The protocol reuses two already-open actions, `double_lift_zebra` and
`double_stretch_zebra`. Only their frame-zero material particles, known
controller trajectories, registered attachment maps, and existing incumbent
physical predictions are permitted. Prefix and future object outcomes remain
unread during this stage. No target or held-v8 artifact is in scope.

Genesis is initialized with `morphs.Nowhere`, then populated with the exact
4,607- or 4,208-particle source roster. Particle order is therefore the source
material identity. Registered attachment particles and their frame-boundary
targets are supplied by the same inverse-distance controller map already present
in the source-input artifact.

Genesis 1.3.3's `Nowhere` entity intentionally starts inactive. Its public
`activate()` helper rejects that same state before it can update the particle
mask, while the upstream emitter path activates particles directly. The frozen
runner follows the emitter semantics: it first activates the complete particle
roster in the solver, then marks the entity forward-active so the public
`set_free()` guard permits the registered fixed-particle mask. The first sealed
attempt stopped at this API mismatch before producing a trajectory. It also
reported Genesis's stability recommendation, so v1 uses 64 base substeps and
128 refinement substeps at 30 Hz; both lie below the reported maximum stable
substep duration.

That first physics attempt also exposed a distinct semantic mismatch before any
object outcome was opened. Genesis implements `free=False` MPM particles as
zero-velocity grid boundaries. Using that flag for moving attachments therefore
erased their actuation velocity, produced no Young's-modulus sensitivity, and
left the source-value gate closed. The amended bridge keeps all particles free
during MPM substeps and overwrites registered attachment positions and
velocities at each 30 Hz frame boundary. This is the only controller-boundary
arm in v1. Domain padding is 0.15 m so the exact source roster remains beyond
Genesis's grid safety margin.

The amended bridge passed all checks except the original 5 micrometre
continuous-translation gate. MPM is evaluated on a fixed Eulerian lattice, so
an arbitrary sub-cell translation changes interpolation weights even when the
domain bounds move with the object. The final frozen probe keeps the strict
5 micrometre threshold for an integer-grid translation and reports an
off-lattice translation separately. The off-lattice discretization error is
capped at 0.0005 m, exactly 0.1 of the registered particle diameter. This
grid-aware change was made from source-physics evidence only; source object
outcomes remained closed.

## Frozen probes

For the first ten action frames, the runner executes:

1. a base driven replay twice;
2. a zero-action replay;
3. a translated-frame replay with translated MPM bounds;
4. a doubled-substep replay;
5. a soft-material replay; and
6. a stiff-material replay.

The gate checks exact repeated arrays, zero-action drift, translation
equivariance, time-step refinement, fixed active-particle identity, finite and
bounded deformation, nondegenerate material sensitivity, frame-zero parity to
the incumbent query, and byte-exact incumbent fallback. Translation is the
only rigid-equivariance probe in v1; no rotational or gradient claim is made.

If any check fails, source-value scoring remains closed and the existing
incumbent archives remain the exact fallback. Passing authorizes only a new,
separately frozen source-value comparison. It does not modify or reinterpret
the confirmed DEFORM DLO2 result.

The retained run passed this gate. Its separately frozen source-value comparison
then failed and preserved exact incumbent fallback; see
[`genesis_mpm_zebra_source_value_v1_result.md`](genesis_mpm_zebra_source_value_v1_result.md).
