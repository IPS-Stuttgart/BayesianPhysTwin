# RGBench ARCSim competence v8 result

## Decision

**Technical failure before simulator invocation.**

The exact frozen implementation commit was
`e83a28aff175ff681d8cbcc66e433e2df5cabc13`. Its first isolated replay
stopped during provenance validation because the protocol named the two built
static libraries at:

- `dependencies/json/lib/libjson.a`;
- `dependencies/alglib/lib/libalglib.a`.

The bound files actually reside in `dependencies/lib/` after the compatibility
build. Their bytes and expected hashes are present, but the frozen relative
paths are wrong. The replay exited after 0.099 s, before ARCSim was invoked.

This result says nothing about ARCSim determinism, control accuracy, runtime, or
cloth accuracy. It identifies a runner/protocol path defect only.

## Evidence

- Remote gate:
  `/home/florianpfaff/results/rgbbench-arcsim-competence-v8-e83a28a/gate.json`
- Gate SHA-256:
  `281b27339155432c16aeb7fe06847a3e80e0c1a6723d9d12923439d2473665d5`
- Replay log SHA-256:
  `4f1dbcda39ac05f124733c74b4a4277769c7d033285b08ff3e074e0586121548`
- Official archive SHA-256:
  `053239c4fbc566228d3f46e8afd3428dc2ffa1c2d18d348af7b1094cd8f5a26e`
- ARCSim executable SHA-256:
  `df8a682bc45c634853a7a79e423740382a43727bb2f2763412309562e37a459b`

The gate records that no point-cloud filename, point-cloud coordinate, source
accuracy outcome, or future object outcome was read.

## Next action

Do not overwrite or rerun v8. Freeze v9 with only the two corrected static
library paths, preserve every numerical threshold and information boundary,
and rerun the same target-free two-replay gate.
