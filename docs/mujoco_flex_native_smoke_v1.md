# MuJoCo volumetric Flex native smoke v1

`scripts/remote/run_mujoco_flex_native_smoke.py` is the first native-execution
gate for the registered `mujoco-flex-v1` backend. It is deliberately synthetic:
passing this gate proves that the pinned native engine, the fixed-identity
adapter, and the portable material-trajectory producer execute together. It does
not establish source value, fresh-object value, calibration, or downstream
Causal4D benefit.

## Frozen runtime

- MuJoCo `3.9.0`, Git tag revision
  `237c17e48539b6c90bf90d3161547cbdcbfaa1e0`.
- CPython 3.10 on Linux x86-64.
- Wheel `mujoco-3.9.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`,
  SHA-256 `c148824d73487fe5ee29c371eff981645f372ccada1f20ea331288323e37c65e`.
- The runner also verifies the installed Python facade, native binding, and
  bundled MuJoCo shared library byte hashes before constructing a scene.

## Frozen synthetic scene

The scene is a three-dimensional `5 x 3 x 3` grid Flex with 45 persistent
vertices and 96 tetrahedra. The nine minimum-x vertices are pinned. A force is
applied to the nine maximum-x vertex bodies in the driven replay; the matched
zero-action replay receives zero force. Gravity and contact are disabled so the
zero-action equilibrium is an exact check rather than a tuned settling phase.

The smoke requires:

1. two complete portable bundles to be byte-identical;
2. zero-action drift at most `1e-12 m`;
3. driven response greater than `1e-4 m`;
4. the response at half the registered Young's modulus to exceed the response
   at twice the registered Young's modulus by at least 25 percent; and
5. fixed compiled topology, material order, finite state, and exact provenance.

## Isolated execution

Download the exact wheel, verify its published digest, create a CPython 3.10
environment, install BayesianPhysTwin from the exact candidate revision, then
run:

```bash
python scripts/remote/run_mujoco_flex_native_smoke.py \
  --wheel /absolute/path/to/mujoco-3.9.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
  --output-dir /new/ordinary/output/directory
```

The output directory must not exist. The runner stages two independent replays,
publishes deterministic portable artifacts, writes
`mujoco-flex-native-smoke.json`, and atomically installs the completed directory.
