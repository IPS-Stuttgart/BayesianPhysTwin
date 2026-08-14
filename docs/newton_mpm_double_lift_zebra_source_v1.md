# Newton MPM source gate v1

## Question

Can the Newton implicit-MPM backend consume a real, already-open PhysTwin
object/action contract and produce a useful continuation under the same point
metrics as the incumbent PhysTwin and MatPhys/Warp replay?

This is deliberately a one-case development mechanism gate. It cannot support
a fresh-data, calibration, PhysWorld-reproduction, or state-of-the-art claim.
No target or held-out protocol artifact is in scope.

## Source case

The frozen case is `double_lift_zebra`, reused from the completed MatPhys
backend real-replay smoke. It is unusually suitable for the first MPM bridge:

- the 4,607-node physical frame-zero state is byte-identical to the released
  3,887 object identities followed by 471 surface and 249 interior points;
- the full 58-frame, 30-point controller trajectory is a separately recorded
  known action;
- the official PhysTwin controller radius and neighbour limit are preserved;
- matched incumbent and MatPhys/Warp six-array physical archives already
  exist; and
- the source role and its limitations are already public.

The configuration and all source SHA-256 identities are frozen in
`configs/sota/newton_mpm_double_lift_zebra_source_v1.json` before a full-horizon
MPM score is produced.

## Mapping

Every physical graph node becomes one Newton material particle in the same
order. For each frame-zero controller point, the official PhysTwin radius and
maximum-neighbour query identifies attached object particles. A material
particle reached by multiple controller points receives their displacement by
normalized inverse-distance weighting. Those particles are kinematic in both
the driven and zero-action MPM runs. The driven run follows the known
controller trajectory; the zero-action run holds the same attachment particles
at frame zero.

This is a testable contact approximation, not a claim that hard attachment is
the correct zebra contact physics.

## Frozen grid and split

The MPM grid crosses four Young's moduli (`25 kPa`, `100 kPa`, `500 kPa`, and
`2 MPa`) with damping values `0.002` and `0.02`. All other MPM settings are
fixed. Candidate selection uses object frames 1--29, the advancement gate uses
frames 30--39, and frames 40--57 remain closed until the gate decision.

Selection minimizes the equal-weight mean of identity-RMSE and Chamfer ratios
against exact persistence. The selected MPM arm must then:

1. beat persistence by at least 5% on the balanced validation score;
2. remain within 10% of the incumbent on each validation metric;
3. keep zero-action drift below 2 mm;
4. replay to a coordinate RMSE at most `1e-7` m;
5. produce non-degenerate but bounded ensemble spread; and
6. preserve finite, fixed-identity, metre-scale physical arrays.

Failure selects a byte-exact copy of the incumbent physical archive. MatPhys is
a matched comparator, not a fallback-selection candidate.

## Information boundary

Preparation separates the known frame-zero geometry and complete controller
action from prefix and future object outcomes. Prediction receives only the
geometry/action artifact. Prefix scoring is performed only after every MPM
candidate and its deterministic replay are sealed. Future scoring is permitted
only after the validation gate passes. Technical failures remain in the fixed
eight-candidate denominator.

Each grid manifest binds the clean Git commit and SHA-256 identities of the
runtime, gate, and CLI modules, plus exact Newton, Warp, NumPy, SciPy, and
Python runtime versions. Candidate records are schema-checked and their
material parameters must match the same-index frozen grid entry. Before any
future file is opened, the prefix result's content ID, selected archive, and
all validation checks are independently re-derived from the sealed grid.

The initial ten-frame feasibility run read frame-zero geometry and controller
motion only. It established finite execution for 4,607 irregular source
particles and 107 attached material particles; it did not inspect an MPM object
motion score.
