# Public Deform360 benchmark comparators

## Scope

This note fixes the published Deform360 numbers that may be used as literature
context for the public-data BayesianPhysTwin study. The study uses measurements
already released by Deform360. It requires neither new physical capture nor
human registration approval.

The primary source is
[Deform360, arXiv:2607.05390v1](https://arxiv.org/abs/2607.05390v1). The public
processing repository is `lhy0807/deform360` at the protocol-pinned revision
`d8522a4403b766aeb387510c04e89032a56fdf35`. Its README states that baseline
training code and pretrained world-model checkpoints are not part of the public
release.

## Published 3-D results

The values below are transcribed from Tables 3--5 of the paper. They are retained
in the paper's own scale; this note does not infer a unit conversion that the
tables do not declare.

| Evaluation regime | Method | Future CD | Future track error |
|---|---|---:|---:|
| Per-episode frame generalization | PGND | 0.073 | 0.073 |
| Per-episode frame generalization | ParticleFormer | 0.044 | 0.041 |
| Per-episode frame generalization | PhysTwin | 0.014 | 0.025 |
| Multi-episode generalization | PGND | 0.130 | 0.144 |
| Multi-episode generalization | ParticleFormer | 0.051 | 0.079 |
| Multi-object generalization | PGND | 0.429 | 0.320 |
| Multi-object generalization | ParticleFormer | 0.038 | 0.048 |

Cosmos is reported only in image-space metrics in the multi-episode and
multi-object tables, so it has no published 3-D CD or track-error value to place
in this table.

## Claim boundary

These rows are not interchangeable. Per-episode PhysTwin fits one episode,
multi-episode methods train on other episodes of the same object, and
multi-object methods test zero-shot object transfer. The registered
BayesianPhysTwin study instead calibrates policies on ten physical objects and
keeps twelve disjoint physical objects sealed while using only a causal prefix.

Consequently, improvement over the registered physical and last-residual
comparators is evidence that the Bayesian update helps under its own frozen
protocol. It is not, by itself, official Table-3, Table-4, or Table-5 parity and
must not be called a Deform360 state-of-the-art result.

A protocol-matched SOTA claim additionally requires:

1. the same object, episode, cutoff, annotation, and metric definitions for every
   method;
2. target-blind execution of each comparator on the same frozen confirmation
   cohort;
3. equal-object aggregation and paired uncertainty intervals;
4. retention of technical failures and exact fallbacks in the denominator; and
5. publication of both positive and negative results without post-confirmation
   retuning.

Because the official public repository does not release the benchmark training
code or checkpoints, a future direct comparator must either obtain the authors'
exact artifacts or transparently preregister a faithful reproduction. Published
table values remain literature context until that protocol-matched comparison
exists.
