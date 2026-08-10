# Deform360 joint-sparse MotionCrafter source plan v5.1

This source-only amendment repairs a temporal-lineage mismatch discovered before
any v5 source forecast, development suffix, score, confirmation payload, or
target outcome was opened.

The archived official-Hub Prob4D products were generated from a previously
locked tactile/contact-centered interval. They retain useful camera-roster,
Prob4D, MotionCrafter, model-set, and stochastic-runtime provenance, but they do
not cover the action-selected 58-frame causal prefix for eight of the ten v5
development objects. Those archived products are therefore rejected as v5
observations rather than treated as technical fallbacks.

The v5.1 plan applies one deterministic rule to every development object:

1. read the exact 58-frame prefix already recorded in the prepared-source
   inventory;
2. select its latest 42 frames;
3. run two 25-frame MotionCrafter windows with eight frames of overlap;
4. retain the frozen three-camera roster and exact Prob4D/MotionCrafter model,
   revision, seed, and decoding configuration;
5. exclude any endpoint-reserved camera from the later likelihood while still
   allowing its outcome-blind provider artifact to be generated; and
6. use only the decoded-uniform Prob4D export in the registered v5 source plan.

The plan consumes released real-world Deform360 RGB videos. It requires no new
measurement and no human approval. Camera identities come from the earlier
target-free roster; temporal ranges come only from the v5 action-window lock.
No future object frame is permitted.

Build the immutable schedule after committing the runner, so the plan can bind
the exact implementation revision and runner digest:

```bash
python scripts/science/materialize_deform360_joint_sparse_motioncrafter_source_plan_v5.py build \
  --execution-lock protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json \
  --prepared-source-inventory /exact/public-source/prepared-source-inventory.json \
  --camera-roster-manifest /exact/frozen/official-hub-motioncrafter-jobs.json \
  --implementation-revision "$(git rev-parse HEAD)" \
  --runner-source scripts/remote/run_deform360_joint_sparse_motioncrafter_source_v5.py \
  --output /durable/source-only/motioncrafter-source-plan.json
```

The runner validates all three bound source files, exact clean repository
revisions, every video digest and byte count, the per-call seed schedule, and
the overlap-window manifest before publishing an integrity-only run report.
Two GPUs may execute disjoint shards. A final unsharded `--resume` pass validates
all 30 existing jobs and writes the complete run report without duplicating
inference.

This amendment is not performance evidence. Its only claim is that the visual
provider now observes the intended public causal prefixes under a reproducible
and outcome-blind contract.
