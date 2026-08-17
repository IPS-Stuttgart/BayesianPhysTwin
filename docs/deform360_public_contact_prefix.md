# Public Deform360 contact-prefix artifact

This path uses measurements already released by Deform360. It does not require
new capture, a robot experiment, or human approval. It is calibration-only and
does not authorize access to confirmation objects or target outcomes.

Published Deform360 numbers and the boundary for any state-of-the-art claim are
recorded separately in
[Public Deform360 benchmark comparators](deform360_public_sota_comparators.md).

The `tactile-axis-map` command binds each tactile gripper group to one released
robot axis. The map must be produced by a locked source-only rule and is
content-addressed. In particular, bimanual stream names are not silently
interpreted as robot-axis identities and cannot be relabeled after seeing an
outcome.

```bash
python scripts/science/materialize_deform360_calibration_factors.py \
  tactile-axis-map \
  --prepared-source-inventory prepared-source-inventory.json \
  --object-id OBJECT \
  --group-to-robot-axis group-to-axis.json \
  --selection-evidence-id SOURCE_ONLY_SHA256 \
  --output tactile-axis-map.json
```

The `public-contact-prefix` command then verifies the inventory-bound
`robot.npz` and `synced_tactile.npy` bytes, reads only the frozen tactile
prefix, and emits:

- `frame-ids.npy`;
- `sensor-names.json` and `contact-episode-ids.json`;
- `tactile-response.npy`, with the released left/right taxels interleaved in
  the same order as the official 768-point gripper geometry;
- `taxel-world-positions-m.npy` from the released robot pose and opening;
- `source-reliability.npy`, fixed to neutral one rather than inferred from a
  PhysTwin residual;
- exact source lineage, a content-addressed manifest, and `SHA256SUMS`.

```bash
python scripts/science/materialize_deform360_calibration_factors.py \
  public-contact-prefix \
  --prepared-source-inventory prepared-source-inventory.json \
  --processed-root /path/to/aligned-calibration-source \
  --tactile-axis-map tactile-axis-map.json \
  --object-id OBJECT \
  --output-dir contact-prefix
```

The contact detector reproduces the public Deform360 convention: use rows
0--11, require at least two positive taxels, retain the first event, and end it
after more than five missing frames. Inactive patience rows are not emitted as
metric observations. If no mapped event occurs, the command publishes a
`support-negative` artifact and exits with status 3. The object remains in the
registered denominator and cannot be replaced.

This artifact intentionally stops before physical-patch prediction and state
linearization. Those inputs are supplied separately to `contact-anchor`. This
keeps association and prior perception reliability independent of the state
innovation, which is processed only once by the existing robust likelihood.
