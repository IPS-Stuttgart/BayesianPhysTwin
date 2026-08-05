# Deform360 official-Hub metric object-carrier smoke result

## Result

The frozen source-only metric object-carrier gate rejected the candidate and
selected exact fallback. SAM2 produced nonempty causal-prefix masks in all
three registered cameras, but the masked MotionCrafter point maps did not
contain enough mutually consistent three-view geometry to construct the
required 128-node carrier.

| Assignment | `019_cam1` candidates | `013_cam1` candidates | `006_cam0` candidates | Three-view carrier nodes |
| --- | ---: | ---: | ---: | ---: |
| Direct | 25 | 15 | 11 | 0 |
| Swapped | 25 | 15 | 11 | 0 |

The direct and swapped tactile-to-gripper hypotheses remained at equal prior
mass. The failure occurred before graph/material association, state update,
or calibration-score access. No contact anchor was authorized.

## Meaning

The preceding tactile metric-gauge result showed that the three cameras can
be mapped into a common metric frame. This smoke shows that metric gauge is
not sufficient: after causal object masking and correlation-aware point-map
reduction, strict three-view material support can still disappear.

This closes only the frozen carrier construction tested here:

- one deterministic point per fixed `8x8` image block;
- at least 16 object-mask pixels and 50% valid point-map coverage per block;
- mutual-nearest-neighbor agreement within 30 mm in all three views;
- equal-weight covariance intersection for unknown cross-view correlation;
- explicit assignment-mixture covariance;
- exactly 128 output nodes required for the physical backend.

It does not show that two-view evidence with calibrated covariance inflation,
an independent depth/tactile identity carrier, or a different preregistered
object association cannot work. Those would be new protocols. The present
lock cannot be weakened after observing this result.

The recorded boundary is therefore:

- `object_carrier_authorized=false`;
- `contact_anchor_authorized=false`;
- exact fallback selected;
- calibration scores unopened;
- confirmation and target payloads unopened;
- no held-v8 access;
- no SOTA claim.

## Reproducibility

The operational run used the already established Deform360 processing
environment (`torch 2.4.0+cu121`) with the exact pinned SAM2 repository and
checkpoint from the lock. This repaired only the runtime dependency path; it
did not change the frozen method.

The original and independent replay produced byte-identical artifacts:

- lock ID: `bab42f816d11b8f13e885698ab86f771c4d666367815fe3250cd0c6c88c89d45`;
- implementation revision: `9dd5c563921d8d25b211edd51e010344a6606b7b`;
- full result artifact ID: `76704e497b90ea28a6f707ea582a35fa06c7dfd6d04e7864cd4315090970fa78`;
- full result SHA-256: `689685e5e9329e0edd6afafaaef9fa964f5b41612abf9243d9fed982fde4c351`;
- carrier NPZ SHA-256: `53778f5bc03441c76e5875bcccbb0aed6c0084659b56477dd69ded7b7fa0c483`.

The full source-only artifacts are preserved at
`/home/florianpfaff/source-only/deform360-metric-object-carrier-smoke-v1-9dd5c563`.

Focused carrier verification passed 9 tests and Ruff lint. The full suite on
the exact remote checkout completed with 2,170 passed, 2 skipped, and 4
environment-specific failures: three video-cadence tests require an FFmpeg
version supporting `-fps_mode`, and one compatibility test expects the NumPy 2
private namespace while this established SAM2 runtime contains NumPy 1.x.
None of those failures exercises the carrier implementation.
