# Adapter replay runtime recovery v1

The original retrospective adapter-control workflow 33751251324 failed before
computing ablation outcomes. Maximum physical replay discrepancies were
0.0112653 mm for DLO4 and 0.0157058 mm for DLO5, above the unchanged 1e-8 m
parity limit. No adapter result is inferred from that technical failure.

Source inspection found a runtime mismatch: the parent `_train_physical`
calls `_seed_everything`, enabling deterministic PyTorch/cuDNN algorithms;
the adapter-only runner called `_setup_torch` without restoring those flags.
The physical rollout functions are otherwise unchanged between the frozen
parent and the failed adapter revision.

This recovery restores the same seed and deterministic flags before model
construction. It changes no checkpoint, data, feature arm, shrinkage rule,
sample-size schedule, evaluation metric, or tolerance. The original protocol
remains byte-identical. A single execution per already-open DLO is allowed in
a new private directory. Both physical and adapter parity must still pass
before any control outcome is computed. A second parity failure remains a
technical failure, not permission to increase the tolerance or resample.

This is retrospective runtime repair, not a new prospective experiment.
No held-v8 or other reserved payload is authorized. No frozen parent artifact
is written. Recovery outcomes remain local/private pending explicit release.
