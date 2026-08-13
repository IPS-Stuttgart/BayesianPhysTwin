# Deform360 v6.1 public-source scoring

## Scope

This stage scores the already sealed 100-record v6.1 candidate panel on the
released Deform360 source suffix. It uses public real-world RGB recordings and
the registered robot stream. It collects no new measurement and requires no
human approval, prompt, mask choice, camera substitution, or outcome-based
replacement.

The upstream observation feeder remains the disjoint MotionCrafter baseline.
Prob4D produced part of the historical provider bundle, but decoded-uniform
Prob4D overlap fusion is unused here and no new MotionCrafter or Prob4D
inference is run.

## Information order

1. Revalidate the exact 100-record candidate panel and its successful protected
   execution receipt.
2. Publish a workflow-bound suffix-opening authorization.
3. Materialize endpoint geometry for exactly the ten registered public source
   object-sessions.
4. Score every candidate on a common B0-defined query roster.
5. Run the unchanged nested source gate once.

No confirmation payload, target outcome, or held-v8 artifact is read by this
stage. A technical failure never becomes a scored loss, cannot authorize a
replacement, and leaves the source gate unevaluated.

## Endpoint carrier

The carrier uses frames `[58, 76)` for scoring and leaves `[76, 81)` unscored.
The two reserved views do not contribute to prediction. Each reserved view must
support at least nine scored cells, every scored frame must have at least one
reserved-view cell, and reconstruction requires at least two nonempty automatic
SAM2 masks per frame. Partial mask support is retained as missing support rather
than filled or silently replaced.

The runtime is Python 3.10 with Torch `2.4.0+cu121` and the exact precompiled
`gsplat==1.4.0+pt24cu121` wheel. Both wheel and embedded CUDA-extension bytes
are hash-locked; JIT compilation and `nvcc` fallback are forbidden.

## Decision boundary

A negative source gate retains B0 and terminates this method honestly. A passing
source gate authorizes only a separately frozen continuation step. It does not
open independent confirmation by itself and cannot support a state-of-the-art
claim without that independent evaluation.
