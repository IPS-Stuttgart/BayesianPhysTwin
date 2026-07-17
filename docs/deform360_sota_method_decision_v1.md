# Deform360 SOTA method decision v1

Date: 2026-07-17

## Decision

The primary route to a stronger-than-published Deform360 multi-episode result is
an automatically registered, object-specific reusable PhysTwin. The method
claim is not a generic physics-neural hybrid. It is:

1. one canonical object graph registered automatically to each initial state;
2. physical parameters pooled across the six fit interactions;
3. a Bayesian parameter ensemble used for predictive uncertainty;
4. outcome-independent admission of simulator response;
5. exact persistence when the response is unsupported;
6. an optional source-trained causal contact-transition schedule, admitted only
   by its separately locked development gate.

This directly addresses the reason Deform360 excludes PhysTwin from its
multi-episode Table 4: manual per-episode registration. Deform360 reports
PhysTwin at `0.014 m` future Chamfer and `0.025 m` future track error in the
per-episode setting, while its multi-episode ParticleFormer reference is
`0.051 m` and `0.079 m`. Those values establish headroom, not an authorized
comparison to our independent protocol.

## Recent competitor boundary

PGRD (arXiv:2607.13451) already occupies the spring-mass plus learned temporal
velocity-residual design. Its public repository is pinned externally at
`e294d96723054f77a1cfdd3c2c052de7b7cd9ce3`. It uses a PTv3 encoder, a
position-conditioned decoder, gated sliding-window temporal refinement, and
multi-step rollout training. The repository contains no license file at the
pinned revision, so its source is not copied into this project. It may be run
externally as a faithful comparator once Deform360 annotations are available.

PointWorld (arXiv:2601.03782) establishes the scale-based 3D point-flow route.
3DPWM (arXiv:2607.00148) establishes explicit point completion plus multi-step
3D dynamics. Neither reports the locked Deform360 multi-episode contract, so
neither can be numerically ordered against this experiment without a new run.

The earlier Bayesian-PhysTwin neural-residual source test is retained as a
negative result. It is not represented as a PGRD reproduction and cannot support
a superiority claim over PGRD.

## Promotion order

The development panel evaluates changes in this order:

1. persistence;
2. single-fit-episode physical selection;
3. pooled reusable PhysTwin;
4. pooled reusable PhysTwin with frozen observable trust;
5. causal onset/release contact transition with the same trust policy;
6. externally run PGRD-style residual comparator when its data adapter is
   available.

The contact-transition arm must beat arm 4, not merely persistence. A learned
residual is never promoted from training loss or from one favorable object. It
must improve held development geometry and uncertainty under the same initial
frame, robot trajectory, particle identities, horizon, and aggregation.

## Claim boundary

The released Deform360 repository currently contains the raw data and annotation
pipeline but not the Table 4 object split, temporal horizon, particle identity
contract, aggregation, baseline checkpoint, or evaluator. The paper source also
describes track error as mean squared error without resolving whether the table
reports mean distance, RMS distance, or squared distance. Its appendix allows
the illustrated `PhysTwin*` upper bound to observe 80% of each testing episode;
that is not the zero-shot multi-episode comparison proposed here.

Consequently, a passing locked panel is an independently preregistered public
benchmark result. A direct state-of-the-art statement is authorized only after
the typed evaluator contract reproduces ParticleFormer's published
`0.051/0.079` row exactly.

References:

- Deform360: https://arxiv.org/abs/2607.05390
- PGRD: https://arxiv.org/abs/2607.13451
- PGRD code: https://github.com/shivanshpatel35/pgrd
- PointWorld: https://arxiv.org/abs/2601.03782
- 3DPWM: https://arxiv.org/abs/2607.00148
