# Newton implicit-MPM backend compatibility v1

## Scope

This opt-in experiment establishes a simulator boundary that is genuinely
different from the PhysTwin/Warp spring family. It runs Newton's implicit
Material Point Method (MPM), retains stable particle identities, and projects a
deterministic subset of those particles into Bayesian-PhysTwin's six-array
physical-rollout contract.

This is a compatibility and mechanism smoke, not a PhysWorld reproduction.
PhysWorld additionally reconstructs and calibrates an MPM twin from video,
generates varied demonstrations, and trains a GNN surrogate. None of those
components is claimed here.

The implementation is additive and does not alter any frozen PhysTwin,
MatPhys, Deform360, PokeFlex, Prob4D, or Causal4D path.

## Boundary

The Newton producer emits:

- `driven_particle_positions_m` with shape `(T,P,3)`;
- `zero_action_particle_positions_m` with the same shape and exact frame-zero
  identity;
- unique persistent `material_query_indices`; and
- one residual-independent `action_support` value per selected query.

The adapter maps those particles to:

- `prediction_m`;
- exact `persistence_m`;
- `driven_readout_m`;
- `zero_action_readout_m`;
- `action_support`; and
- `frame_zero_points_m`.

Positions are in metres in the producer-declared world frame. Query index `n`
denotes the same MPM material particle at every time. The portable
`physical_rollout_v1` validator is simulator-neutral and accepts any frame
count of at least two. The registered smoke emits 76 frames so its output also
passes the current fixed-length Deform360 physical-array consumer.

## Installation and commands

Newton is optional and does not enter the default dependency closure:

```bash
python -m pip install -e '.[mpm]'
```

Run the registered synthetic GPU smoke:

```bash
bpt experiment run materialize-newton-mpm-backend smoke \
  /path/to/output \
  --device cuda:0
```

Adapt an independently produced Newton particle archive:

```bash
bpt experiment run materialize-newton-mpm-backend materialize \
  raw-particle-rollout.npz \
  newton-runtime.json \
  /path/to/output
```

Validate the complete bundle and rederive every physical query array:

```bash
bpt experiment run materialize-newton-mpm-backend validate /path/to/output
```

The output bundle is self-contained and content addressed. It contains the
raw particle rollout, runtime manifest, deterministic physical archive,
artifact manifest, and checksums. Validation rejects changed units, coordinate
metadata, frame-zero identity, duplicate or out-of-range query identities,
non-finite values, roster changes, or any mutated source/output bytes.

## Synthetic smoke

The registered smoke uses a 0.30 m by 0.05 m by 0.05 m elastic beam with its left
end fixed and its right end displaced by 0.025 m. Driven and zero-action
rollouts are independently simulated. Query identities are selected from
frame zero by deterministic farthest-point sampling; action support is derived
only from the simulated driven-minus-zero response.

The smoke is deliberately simple. It answers these implementation questions:

1. Can Newton/Warp execute implicit MPM on the available CUDA runtime?
2. Are MPM material identities stable enough for a portable query trajectory?
3. Are metre units, frame-zero identity, driven/zero controls, and query order
   represented without a spring-graph assumption?
4. Can the resulting archive enter the existing Bayesian-PhysTwin physical
   consumer unchanged?
5. Is an identical replay byte reproducible on the same runtime?

It does not answer whether an MPM twin can be reconstructed or calibrated from
real observations, whether it improves held-out prediction, or whether MPM
uncertainty is calibrated.

## Advancement gate

Do not run a fresh target evaluation from this smoke. The next source-only
stage is one already-open development object with a frozen mapping from its
geometry and registered action/contact to MPM particles and boundaries. Advance
only if:

1. frame-zero material queries align in metres and in the registered frame;
2. zero-action and repeated driven replay establish the numerical noise floor;
3. the MPM mean beats or complements the incumbent on a disjoint source prefix;
4. a parameter ensemble produces non-degenerate, plausible predictive spread;
5. failures fall back through the existing guarded physical-backend selector;
6. no future observations or target outcomes enter construction or selection.

Until those gates pass, Newton MPM is an available experimental producer, not
an empirically validated backend for the Bayesian-PhysTwin paper.
