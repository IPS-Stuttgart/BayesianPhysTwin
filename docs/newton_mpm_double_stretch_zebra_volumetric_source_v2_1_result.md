# Volumetric Newton MPM source gate v2.1 result

## Decision

**FAIL / retain the incumbent physical backend.**

The corrected v2.1 protocol completed all eight frozen candidates on the
already-open public `double_stretch_zebra` source case. The selected Newton arm
improved the balanced validation score over exact persistence by `19.56%`, but
failed both non-regression gates against the incumbent physical rollout.
Consequently, the selected archive is the incumbent source archive byte for
byte and source-future scoring is not authorized.

This is evidence about the frozen mass-preserving convex-hull particleization,
inverse-distance query readout, and compliant finite-mass contact adapter. It
is not a fresh evaluation, a PhysWorld reproduction, a calibration result, or
evidence that Newton or MPM as model families are inadequate.

## Frozen execution

- implementation commit: `add70cf0a31e8af6b5a48cabb2a855e5eaea0c7b`;
- protocol SHA-256:
  `9c7c48b5e71f4141f8ae711f19824f3431a6430b4d090a09aa425e8a0699ab6e`;
- Newton / Warp / NumPy / SciPy: `1.5.0 / 1.16.0 / 2.2.6 / 1.18.0`;
- device: NVIDIA RTX 6000 Ada Generation;
- candidates completed: `8/8`;
- selected Newton candidate: `5 kPa`, damping `0.02`;
- deterministic replay RMSE: `0.0 m`;
- maximum zero-action drift: `0.0 m`;
- final ensemble spread: `0.570 mm`; and
- target or held-out artifacts read: `false`; and
- source-future outcome artifact supplied to the scorer: `false`.

The custody preparer did read the full already-open public source payload once
to create physically separate prediction, prefix, and future files. The
outcome-blind grid read only the prediction file, and the gate scorer read only
the prefix file.

The following table reports the separately frozen validation split in
millimetres.

| Method | Identity coordinate RMSE | Symmetric Chamfer |
|---|---:|---:|
| Persistence | 19.054 | 15.332 |
| PhysTwin incumbent | 5.301 | 4.981 |
| MatPhys/Warp comparator | 4.732 | 4.307 |
| Newton volumetric v2.1 | 14.403 | 13.076 |

The selected Newton arm improves identity RMSE by `24.41%` and Chamfer by
`14.72%` relative to persistence. It regresses by `171.70%` and `162.52%`,
respectively, relative to the incumbent. Its balanced persistence ratio is
`0.80436`, so the persistence-improvement gate passes; both incumbent
non-regression gates fail.

## Interpretation

V2.1 resolves the v1 representation defects sufficiently to retain a stable,
deterministic, noncollapsed eight-arm ensemble and to beat a no-dynamics
baseline. It does not recover the accuracy of the existing source-trained
backends. The source result therefore closes this exact direct Newton adapter
as a replacement backend; it does not justify a larger parameter sweep or an
independent-object evaluation.

The useful positive result is infrastructural: BayesianPhysTwin can run and
gate a volumetric Newton backend with a separate material/query state, fixed
total mass, compliant contact, strict provenance, and byte-exact fallback.
Further Newton work must introduce a genuinely new source-independent
mechanism, such as calibrated state initialization or measured contact
boundary conditions, rather than retuning this opened case.

## Evidence

Compact content-addressed artifacts are in
`results/sota/diagnostics/newton_mpm_double_stretch_zebra_volumetric_source_v2_1/`:

- `source-custody.json`: SHA-256
  `2c4d2b0b7d2a0a87341f4fe29e6459e1d0b26e6abb06a13ea662634e9e379a44`;
- `newton-grid.json`: SHA-256
  `12fa7906cc77ae5d620e8f79e13590b5d619fb4664fb22dc4701068c761f4aa3`;
  and
- `prefix-result.json`: SHA-256
  `8c9a05b6dcdd54ab8a0ce1e468257e36dd7702f49186cd622f043a31dc282373`.

The prefix artifact records `future_scoring_authorized=false`,
`future_outcomes_read=false`, and `selection=exact_incumbent_fallback`. No
v2.1 future-result artifact exists.
