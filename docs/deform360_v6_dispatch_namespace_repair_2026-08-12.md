# Deform360 v6 dispatcher namespace repair

Date: **2026-08-12**
Status: **pre-source-prediction technical repair; all suffix, confirmation, and target data remain closed**

## Retained failure

Protected-main run `31540957671` built and probed both frozen CUDA runtimes
successfully. Before any physical manifest or source prediction was produced,
the source launcher stopped with:

```text
v6 selector runtime repair identity changed
```

The compact artifact is `9120751619`, with artifact digest
`sha256:34125b0ef1c3cbbbe8e65c5917f77a05745019d7cd4dfb65c57cbfdc7b9b6555`
and receipt ID
`4f292b6fa74f9895a04bb30793cdc86f457e88cf9cf0d3d3c59dd6b78cf6f61f`.
It records zero physical manifests, zero source-prediction seals, and false for
every suffix, confirmation, and target-access flag.

## Root cause

The frame-zero dispatcher used the generic local shell name `REPAIR_ID` for
its own immutable runtime identity. The existing selector wrapper also exports
`REPAIR_ID` for its content-addressed amendment before invoking the dispatcher.
The dispatcher replaced that inherited value, so the selector verifier compared
its amendment against the unrelated frame-zero identity and failed closed.

## Repair

The dispatcher-local constant is renamed to
`FRAME_ZERO_DISPATCH_REPAIR_ID`. Its value and marker bytes are unchanged. The
workflow now probes that an inherited selector `REPAIR_ID` crosses the
dispatcher unchanged, verifies exact amendment and dispatcher digests before
installation, and records the namespace repair in every bounded receipt.

## Frozen scope

This repair changes no object, episode, camera panel, input frame, selector,
model, physical algorithm, candidate mean, candidate covariance, horizon, loss,
fallback, suffix rule, or target rule. It creates no replacement execution on a
pull request. A reviewed protected-main merge may authorize exactly the normal
registered source execution, which must still yield ten complete physical
manifests and 100 immutable source-prediction seals before any suffix access.
