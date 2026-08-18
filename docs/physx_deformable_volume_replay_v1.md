# PhysX deformable-volume replay v1

`PhysXDeformableVolumeReplayV1` closes the execution boundary for the already
registered experimental `physx-fem-v1` backend without adding PhysX or CUDA as a
BayesianPhysTwin dependency.

## Authoritative state surface

The adapter is deliberately restricted to the **simulation tetrahedral mesh**.
At NVIDIA-Omniverse/PhysX revision
`3ca45ad36e9755f7c8c5bea9f7c57d308d9f0c54`,
`PxDeformableVolume::getSimPositionInvMassBufferD()` returns a device buffer with
one `PxVec4` per simulation-mesh vertex. The first three components are the
simulation-node position and the fourth is inverse mass. The buffer size is
`getSimulationMesh()->getNbVertices() * sizeof(PxVec4)`.

This is not interchangeable with
`PxDeformableVolume::getPositionInvMassBufferD()`, which addresses the collision
mesh. Render, collision, and skinning vertices are not accepted as material
identity for `physx-fem-v1`.

PhysX documents that the simulation-position buffer may be read only after all
PhysX tasks have finished (for example after `PxScene::fetchResults()` returns).
The SDK exposes a device pointer; host transfer remains the caller's
responsibility.

## Replay usage

A producer-side scene should own the PhysX scene, deformable volume, CUDA stream
or context, and a persistent host mirror of the simulation-position/inverse-mass
buffer. The replay adapter receives only four narrow facts:

```python
from bayesian_phystwin.physx_deformable_volume_replay_v1 import (
    PhysXDeformableVolumeReplayV1,
)

replay = PhysXDeformableVolumeReplayV1(
    simulation_vertex_count=simulation_mesh_vertex_count,
    read_sim_position_inv_mass_callback=read_host_sim_position_inv_mass,
    advance_callback=advance_one_output_step,
    synchronize_callback=finish_physx_and_copy_sim_state_to_host,
    context=scene,
)
```

The synchronization callback is mandatory. It must make the exact simulation
state visible in host memory before `read_sim_position_inv_mass_callback()` is
called. The read callback must return floating data with shape `(N, 4)` in the
fixed simulation-mesh vertex order. The adapter rejects count drift, a different
buffer layout, integer or non-finite state, and negative inverse masses, then
returns a copied contiguous `(N, 3)` XYZ array.

`produce_material_trajectory_backend(...)` still constructs the driven and
zero-action arms independently, captures frame zero before control, synchronizes
before every state capture, and publishes the unchanged `physical_rollout_v1`
artifact consumed by BayesianPhysTwin, Prob4D, and Causal4D.

## Scientific boundary

This adapter establishes an explicit state-access and synchronization contract.
It does **not** advance `physx-fem-v1` beyond the repository's registered adapter
stage and does not establish native-scene fidelity, calibrated uncertainty,
parameter identifiability, fresh-object transfer, Prob4D benefit, Causal4D
intervention benefit, deployment safety, or state of the art.

Promotion still requires a pinned native PhysX source replay and the common
source-only backend qualification and guarded non-harm gates. Until then PhysX
remains experimental and exact fallback remains authoritative.
