# DLO-Lab Separation Headroom v1 Launch Result

The sole v1 launch is terminally retained as a **zero-science runtime-preflight
failure**. The registered root was created at source revision
`6c3d6a78b183ae28ecb93f5ad2be209424aa339e`, but no lock, worker claim,
native bundle, reward, or result artifact was published. The root has zero
entries and remains untouched.

The exact invocation set `PYOPENGL_PLATFORM=osmesa` and `PYTHONPATH=src` but
omitted the complete registered CPU/OSMesa environment. Re-evaluating the pure
`runtime()` preflight under that invocation environment returns:

```text
ValueError: registered CPU/software-rendering environment required
```

This is not evidence about DLO-Lab separation dynamics, action headroom, or the
proposed decision rule. It accessed no protected data and started zero native
worlds. The v1 root will not be reused. Any successor must preserve the v1
physics, worlds, actions, and gates while validating runtime and source identity
before consuming a separately registered write-once attempt ledger.
