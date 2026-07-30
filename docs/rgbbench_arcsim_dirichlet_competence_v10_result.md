# RGBench ARCSim Dirichlet Competence v10 Result

## Decision

**Gate failed. No RGBench point-cloud outcome was opened.**

Two isolated full-resolution replays completed in about 27.2 seconds each,
preserved all 9,865 vertex identities, remained finite, moved nontrivially, and
were byte-identical. The maximum pin-target error nevertheless remained:

```text
21.7585 mm
```

against the frozen 0.01 mm limit.

## Target-Free Diagnosis

The emitted scene contains `kinematic: true` on both known handles, and the
patched source applies those handles after every substep. The unchanged error
comes from an earlier initialization boundary: ARCSim first equilibrates the
cloth under gravity, then lazily initializes each handle reference from the
already shifted node. The later exact projection therefore follows the known
motion around a gravity-shifted reference rather than around the released
frame-zero actuator attachment.

This diagnosis uses only the scene, simulator source, known actuator targets,
and simulated pin positions. It does not use a segmented point-cloud filename,
coordinate, or accuracy outcome.

## Frozen Checks

| Check | Result |
|---|---:|
| Both replays complete | Pass |
| Byte-identical final arrays | Pass |
| Finite vertices | Pass |
| 9,865-node identity contract | Pass |
| Expected 10-step readout | Pass |
| Nontrivial mean motion | Pass, 56.179 mm |
| Runtime at most 600 s | Pass, 27.170 s max |
| Pin error at most 0.01 mm | **Fail, 21.758 mm** |

## Provenance

- Frozen implementation: `e0120684aa2af253e1a5a9fcf9ffae2372a0f8b1`
- Remote result:
  `/home/florianpfaff/results/rgbbench-arcsim-dirichlet-competence-v10-e012068`
- Gate SHA-256:
  `ca78c92df8ff510e3ae2c6f2f164c6443eb123b357c648ea9b691dda66ea2065`
- Replay array SHA-256, both runs:
  `7b9aad20ce071b2a875cf0d2b3d71f9ae5d405b78c273bf2990007d1132cc424`

V10 closes this initialization order. A successor may initialize the declared
Dirichlet references before relaxation and then rerun the identical target-free
gate. No post-gate stiffness tuning or point-cloud scoring is authorized.
