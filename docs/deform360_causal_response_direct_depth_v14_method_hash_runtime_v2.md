# V14 Method-Hash Runtime V2

## Trigger

The frozen V14 method protocol registers the namespaced checksum
`4bb4133a...`. Before the first source admission, an operational audit found
that both the frozen admission runner and frozen source-finalizer runner
recomputed an unnamespaced checksum, `23be9dfe...`, and would therefore reject
the unchanged registered protocol.

This is a runner defect, not a method change. At discovery:

- no source admission artifact existed;
- no twelve-case source lock existed;
- no prefix scan had started;
- no source outcome had been read; and
- no target or held-v8 boundary had been crossed.

## Amendment

The V2 wrappers dynamically load the otherwise unchanged frozen parent runner
and replace only its `_canonical_config_sha256` function with the namespace
already used to create and test the method protocol:

```text
deform360-causal-response-direct-depth-v14-protocol\0
```

The method JSON, estimator, gates, artifact schemas, source ordering, and all
numerical settings remain byte-for-byte unchanged. Each wrapper first validates
the amendment, method protocol, admission prelock, source-finalizer protocol,
parent runners, wrappers, and runtime module by exact SHA-256.

## Operators

Use the admission wrapper in place of the original admission script:

```bash
python scripts/remote/run_deform360_causal_response_direct_depth_v14_admission_runtime_v2.py \
  --method-hash-runtime-v2 \
  configs/sota/deform360_causal_response_direct_depth_v14_method_hash_runtime_v2.json \
  <the original admission arguments>
```

Use the source-finalizer wrapper in place of the original finalizer script:

```bash
python scripts/remote/finalize_deform360_causal_response_direct_depth_v14_source_runtime_v2.py \
  --method-hash-runtime-v2 \
  configs/sota/deform360_causal_response_direct_depth_v14_method_hash_runtime_v2.json \
  <the original source-finalizer arguments>
```

The wrapper-only argument is removed before the frozen parent parser runs. Any
change to a bound parent or implementation file fails closed.
