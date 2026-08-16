# Genesis MPM backend v1

This opt-in backend connects the released Genesis elastic Material Point Method
(MPM) runtime to Bayesian-PhysTwin's engine-neutral `physical_rollout_v1`
artifact. It is a compatibility layer, not a new default simulator.

## Maturity boundary

The v1 milestone requires:

- a pinned `genesis-world==1.2.2` optional runtime;
- fixed material-particle identities across driven and zero-action rollouts;
- metres, seconds, and a right-handed z-up world frame;
- a deterministic six-array physical rollout artifact;
- content-addressed runtime, raw-rollout, query-map, and output provenance;
- strict rejection of changed units, frame-zero geometry, query identities, or
  archive bytes;
- an exact incumbent fallback outside the backend artifact; and
- a native synthetic smoke before any source-data competence gate.

The smoke uses an elastic beam attached to two rigid grippers with Genesis'
compliant particle constraints. One gripper follows a known displacement while
the matched zero-action run leaves it stationary. This deliberately exercises
volumetric MPM and compliant attachment behavior that the Newton direct adapter
did not represent.

The registered smoke also rejects a maximum particle response larger than
three times the commanded displacement. This catches numerically finite but
mechanically implausible attachment blow-ups before an artifact can be sealed.

## Commands

Install the optional engine in an isolated environment:

```bash
pip install -e '.[genesis-mpm]'
```

Run and seal the synthetic smoke:

```bash
CUDA_VISIBLE_DEVICES=0 bpt experiment run materialize-genesis-mpm-backend \
  smoke /tmp/genesis-mpm-smoke --backend gpu
```

Validate an existing bundle without importing Genesis:

```bash
bpt experiment run materialize-genesis-mpm-backend \
  validate /tmp/genesis-mpm-smoke
```

The `materialize` subcommand accepts a separately generated, fixed-identity raw
particle archive and runtime manifest. The portable validator has only NumPy as
a runtime dependency.

## Claim boundary

Passing the synthetic smoke establishes only that Genesis can produce a
custody-checked Bayesian-PhysTwin physical artifact while preserving material
identity. It does not reproduce PhysWorld or DeformMaster, validate a real
deformable twin, or justify a point-accuracy or uncertainty claim. Any source
gate must be separately frozen and must retain exact incumbent fallback.
