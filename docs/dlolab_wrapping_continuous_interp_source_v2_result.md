# DLO-Lab wrapping continuous interpolation source result v2

## Status

**Complete source result; prospective gate failed.**

The one registered attempt ran under frozen revision
`7683e29dcc1b2525ba5f8d35a83e0c0f66c1e201`. Its runtime preflight passed,
all four prefix batches sealed, and the pre-future gate passed before any task
future was generated. All 32 fresh continuous-material worlds then completed
ordinary native QA with zero technical failures or replacements.

Continuous posterior expected utility improved mean native reward over the
continuous-prior best fixed action by `0.0208874` (paired 95% world-bootstrap
interval `[0.0141595, 0.0280749]`) and captured `55.34%` of available oracle
headroom. It harmed zero worlds beyond the frozen numerical margin.

The denser continuous model did not improve over the simpler nine-particle
Bayesian controller:

| Comparison for continuous Bayes | Mean reward difference | Paired 95% CI |
| --- | ---: | ---: |
| best fixed action | `+0.0208874` | `[+0.0141595, +0.0280749]` |
| finite-particle Bayes | `-0.0008932` | `[-0.0044867, +0.0030197]` |
| continuous MAP | `+0.0008413` | `[-0.0079678, +0.0113841]` |
| ignored shared bias | `+0.0038139` | `[-0.0057830, +0.0153980]` |

The source gate therefore failed its finite-Bayes, MAP, and paired-confidence
requirements. No successor is automatically authorized.

## Interpretation

This result extends the positive decision-value signal beyond the nine source
materials: Bayesian action selection retains a clear advantage over fixed control
on off-grid continuous materials. The gain is not an interpolation gain, however.
The original nine-particle Bayesian controller is marginally better on average and
the difference is unresolved. A compact physical hypothesis bank is sufficient for
this task at the registered sensor quality; densifying it to an 81-point bilinear
quadrature does not add measurable value.

The correctly correlated likelihood also beats the arm that ignores shared sensor
bias in mean reward, but its paired interval crosses zero. This is mechanism
evidence, not confirmation of a covariance-model advantage.

The result supports a scoped paper contribution around calibrated Bayesian
decision-making under material uncertainty and exact fallback. It does not support
a claim that continuous material interpolation improves the controller, identify
physical material parameters, establish benchmark SOTA, or transfer to a robot.

## Evidence identities

| Artifact | Identity |
| --- | --- |
| Frozen source commit | `7683e29dcc1b2525ba5f8d35a83e0c0f66c1e201` |
| Runtime preflight | `e5aeac42cd8a4bd0b402650e3f28ffc5bc470f4b8a16760ce4748706054f16ab` |
| Study lock | `c787cd3000f6f64d5478d800ceed5209ff1490cfd68aadf19fdb543c1cfb101b` |
| Decision barrier | `5b9ffad42827d61a12a5d16379ab5644e85479b736035e7db22e26b690f9de50` |
| Generation seal | `28119ec7e65eb0fbfdac0b99613a51ecefe4b038b01d97e12f67941e83f3a1ca` |
| Result | `2f95e41f51753881cdfe8ca77774cdd60c5d1f92a0552d5570e7299a3244b5bf` |

The second implementation reconstructs all decisions, 32 native reward vectors,
equal-world aggregation, paired intervals, and gates. It is not independent human
review. The complete simulator tree remains under
`/home/fpfaff/source-only/dlolab-wrapping-continuous-interp-source-v2`.
