# V14 Pre-Lock Physical Carrier

## Purpose

V14 needs a frame-zero physical prediction, graph basis, and action-support
field before it can construct the adaptive camera carrier. The adaptive
carrier is itself an input to source admissibility, so requiring an existing
source lock to generate the physical prediction would be circular.

The dedicated pre-lock path removes that circularity without weakening source
admission.

## Frozen Inputs

The physical-prelock protocol binds:

- the unchanged V14 method protocol and source staging queue;
- the exact 12 prefix-geometry bundles at queue ranks 3 through 14;
- the runtime-v1 bundle at rank 3 and runtime-v2 bundles at ranks 4 through 14;
- the exact physical artifact module, automatic-twin wrapper, and physical
  runner bytes;
- the 384-node automatic-twin cap, rank-eight material readout basis, and
  76-frame known-action horizon.

The 12 geometry bundles contain 353 to 2,547 frame-zero material points and
11 or 12 complete cameras. Their previously computed independent validation
digest is
`a330fc1fd5885bc99ed430e98ed47e385018c01f8c1b0da9c08249c020171704`.

## Information Boundary

The physical path reads only:

1. frame-zero object geometry and color;
2. the released 76-frame robot action;
3. frozen numerical configuration and source provenance.

It does not read prefix object response, tactile measurements, future RGB-D,
tracks, identity targets, Chamfer targets, manual trajectories, source
outcomes, target artifacts, or held-v8 artifacts and processes.

Sealed outputs retain only queue rank and cryptographic object/case hashes.
Plaintext object and episode identities are used only to locate released
source files and are not retained in the physical artifact.

## Numerical Path

For an admissible automatic twin, the runner performs official driven and
zero-action Warp rollouts. It exports:

- physical and exact-persistence material trajectories;
- driven and zero-action readouts;
- action support;
- an orthonormal rank-eight graph basis.

If automatic frame-zero twin construction is inadmissible, Warp is not run
and the physical prediction falls back exactly to persistence. Because action
support is then zero, the adaptive carrier must abstain; the case cannot enter
the source lock.

## Prospective Sequence

```text
exact prefix geometry
-> pre-lock physical prediction
-> frame-zero adaptive carrier
-> hash-only source preflight
-> first 12 admitted cases
-> immutable source lock
-> prediction seals
-> source outcomes and registered gates
```

Pre-lock rejection is not a selected case. Once a case has a prediction,
retained technical failure, or outcome disposition, it cannot be replaced.

## Claim Boundary

This path is source-admission infrastructure, not real-data evidence. It
neither changes V14's registered update nor authorizes opening any outcome.
Its purpose is to make the prospective procedure executable while preserving
the original causal and calibration boundaries.
