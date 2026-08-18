# Official MatPhys producer v1

## Purpose

The guarded MatPhys backend previously began after spring prediction and Warp
replay. That was enough to select a sealed candidate, but it did not bind the
official MatPhys checkpoint, semantic preprocessing chain, parameter export,
and fixed-identity replay into one producer artifact.

`matphys_official_producer_v1` closes that interface gap. It converts one strict
official MatPhys/PhysTwin replay export into:

- `matphys-candidate-physical.npz`;
- `matphys-identity-replay-physical.npz`;
- a content-addressed producer record; and
- in causal mode only, the existing guarded `matphys-proposal.json`.

The two physical archives use the unchanged six-array
`physical_rollout_v1` contract. They can therefore enter the same Bayesian
belief and evaluation code as PhysTwin or another registered physical backend.

## Two information regimes

The producer refuses to conflate the published MatPhys benchmark recipe with a
deployable causal predictor.

### `published-per-case-parity-v1`

This mode permits target-specific checkpoint training and always declares the
target fitting prefix. It is the appropriate control for reproducing the
official per-case MatPhys recipe under the published train/future split. The
artifact separately records whether the checkpoint itself included the target,
and records `published_benchmark_control_only=true` and
`causal_backend_eligible=false`. It never emits a causal proposal.

This mode may support a fair point-metric parity comparison. It cannot support
target-transfer, online-abduction, Causal4D, or deployment claims.

### `causal-prefix-transfer-v1`

This mode requires the target object to be absent from the checkpoint-training
objects. Target observations may be used only inside the declared fitting
prefix, whose stop must not cross the registered future boundary. The producer
emits the existing MatPhys proposal manifest, which must still pass a disjoint
prefix gate before its trajectory can replace the incumbent.

Producer eligibility is not model acceptance. Rejection by the downstream gate
still copies the incumbent physical archive byte for byte.

## Replay input

The upstream MatPhys environment exports one deterministic, no-pickle NPZ with
exactly these arrays:

| Array | Shape | Meaning |
| --- | --- | --- |
| `candidate_driven_state_m` | `(T,S,3)` | Full fixed-order Warp state under the MatPhys parameters and registered action |
| `candidate_zero_action_state_m` | `(T,S,3)` | Fresh MatPhys-parameter replay with zero action |
| `identity_driven_state_m` | `(T,S,3)` | Fresh zero-overlay/identity-parameter replay under the action |
| `identity_zero_action_state_m` | `(T,S,3)` | Fresh zero-overlay/identity-parameter replay with zero action |
| `material_query_indices` | `(N,)` | Unique fixed material-state indices exposed downstream |
| `action_support` | `(N,)` | Residual-independent physical action support in `[0,1]` |
| `frame_indices` | `(T,)` | Nonnegative, strictly increasing source frame identities |

All four state arrays must have the same floating dtype and exact frame-zero
state. Positions are metres in `right-handed-z-up-world-v1`. Query identities,
topology, dtype, frame order, and frame zero may not change between arms.

An official runner can write the input without importing Torch or Warp into the
BayesianPhysTwin process:

```python
from bayesian_phystwin.matphys_official_producer_v1 import (
    write_matphys_official_replay_input,
)

write_matphys_official_replay_input(
    "matphys-official-replay-input.npz",
    {
        "candidate_driven_state_m": candidate_driven,
        "candidate_zero_action_state_m": candidate_zero,
        "identity_driven_state_m": identity_driven,
        "identity_zero_action_state_m": identity_zero,
        "material_query_indices": query_indices,
        "action_support": action_support,
        "frame_indices": frame_indices,
    },
)
```

The four trajectories must come from fresh replays. Resetting one hidden Warp
state with retained solver state is not an admissible substitute.

## Bound official pipeline

The producer requires an exact digest for every declared official MatPhys
component:

- video material encoder;
- DINO feature lift;
- part segmentation;
- material distribution;
- GPT physics prior;
- per-edge spring field; and
- collision and damping parameters.

It separately binds the MatPhys and PhysTwin repository revisions, model
checkpoint, complete candidate parameter export, extracted no-pickle spring
field, zero-overlay identity-parameter artifact, source/config artifacts,
training-object roster, target fitting range, full frame order, and replay
archive. The component digests describe custody; they do not imply that each
component improves prediction.

## Command

```bash
bpt experiment run materialize-matphys-official-producer build \
  matphys-official-replay-input.npz \
  checkpoint.pth \
  spring-field.npy \
  candidate-parameters.pth \
  identity-parameters.json \
  output/matphys-producer \
  --mode causal-prefix-transfer-v1 \
  --source-revision <exact-matphys-revision> \
  --simulator-revision <exact-phystwin-revision> \
  --case-id <case> \
  --target-object-id <object> \
  --checkpoint-training-object-id <source-object> \
  --target-fit-start <first-prefix-frame> \
  --target-fit-stop <prefix-stop-exclusive> \
  --future-frame-start <first-future-frame> \
  --proposal-strength 1 \
  --pipeline-component-artifacts component-digests.json \
  --source-artifacts source-digests.json
```

Validate the published bundle without requiring the upstream files:

```bash
bpt experiment run materialize-matphys-official-producer validate \
  output/matphys-producer
```

Add `--verify-sources` while the exact checkpoint, spring field, and replay
input remain available. This rehashes the sources and rederives every output
array.

## Current support level

This change advances MatPhys from a consumer-only guarded archive interface to
a producer-attested official checkpoint/parameter/replay interface. Contract,
custody, mode-separation, and synthetic end-to-end tests are implemented.

No new real-data MatPhys result is created by this interface. The next evidence
step remains one already-open development case that executes the complete
official pipeline, checks the published per-case parity control, and then tests
a separately trained target-excluded causal arm. A fresh or sealed cohort must
not be opened merely because the adapter now exists.
