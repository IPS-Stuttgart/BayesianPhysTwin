# Deform360 Official-Warp Source Gate V1

This milestone records the preregistered source-only admission test for the
official PhysTwin Warp simulator. It was run after the replication protocol was
locked and before any media from the six selected replication objects was
accessed.

## Information Boundary

Only quality-passing 001-rope source episodes 0, 3, 4, 5, and 8 were read.
Episodes 1, 2, 6, 7, and 9 were forbidden. The exhausted pilot target, episode
6, was not read. No selected replication-object media or target outcome was
read.

The run used:

- replication lock commit 3c40eee;
- gate implementation commit ed32fa1;
- official PhysTwin commit 2b6630528141b9cba5a7677c8b88b2129b4a8390;
- the official spring_mass_warp.py simulator;
- deterministic spring-force accumulation;
- a fixed 200-candidate source-only parameter grid;
- a 21-node public rope graph initialized at the end of the six-frame contact
  prefix.

This is an official-Warp backend feasibility result. It is not a dense
reconstruction of a PhysTwin and is not itself a Bayesian-PhysTwin result.

## Result

The gate passed.

| Quantity | Observed | Required |
| --- | ---: | ---: |
| Pooled source Chamfer improvement over persistence | 21.83% | at least 5% |
| Leave-one-source win fraction | 3/5, 60% | at least 60% |
| Leave-one-source mean Chamfer | 32.40 mm | below 39.65 mm persistence |
| Repeat-rollout RMSE | 0.000 mm | at most 0.100 mm |
| Maximum selected p99 edge strain | 12.23% | at most 50% |
| Nonfinite selected states | 0 | 0 |

The pooled candidate selected source-shared stretch, bend, and controller
stiffness values of 1000 and ground friction 0.3. Its pooled mean Chamfer was
30.99 mm, compared with 39.65 mm for persistence.

The leave-one-source wins were episodes 0, 4, and 8. Episodes 3 and 5 remained
negative. The gate therefore clears the preregistered threshold exactly rather
than establishing a uniformly superior backend.

The matched pooling control was selected source-only before target use. The
pooled candidate is 115. Independent single-source fitting selects candidates
197, 112, 29, 117, and 98 for source episodes 0, 3, 4, 5, and 8 respectively.
All six unique candidate identities and parameters are sealed in the pooling
control artifact; target errors cannot choose among them.

## Artifacts

The JSON contains the complete candidate table, leave-one-source folds,
information boundary, numerical audit, official source hash, and internal
result checksum. The NPZ contains two independent selected-parameter rollouts
per source episode.

The JSON preserves the original server path of the prediction archive. The
co-located NPZ is the archived copy and is bound by the same SHA-256 recorded
inside the JSON and in artifact-manifest.json.
