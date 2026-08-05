# PokeFlex Fresh12 Scale Headroom V1

## Status

This is a post-open diagnostic on the already reported fresh12 cohort. It does
not alter the sealed prospective result, authorize a new claim on these takes,
or select a replacement scale for them.

The audit revalidated every staged source manifest, prediction seal, prediction
barrier, and registered result. Multiplier zero reproduced the released
checkpoint score exactly, and multiplier one reproduced the sealed Bayesian
candidate exactly.

## Uniform-scale result

The sealed correction has effective scale `0.125`. The audit rescales only that
already sealed correction field; it does not rerun registration or use target
geometry to form a prediction.

| Multiplier | Effective scale | CD-UL1 (mm) | Improvement | Wins/ties/losses | Worst object |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.0000 | 4.9397 | 0.000% | 0/12/0 | 0.000% |
| 0.5 | 0.0625 | 4.9089 | 0.623% | 11/1/0 | 0.000% |
| **1.0 (sealed)** | **0.1250** | **4.8881** | **1.043%** | **11/1/0** | **0.000%** |
| 1.5 | 0.1875 | 4.8755 | 1.299% | 9/1/2 | -0.980% |
| 2.0 | 0.2500 | 4.8696 | 1.418% | 9/1/2 | -4.216% |
| 3.0 | 0.3750 | 4.8786 | 1.236% | 8/1/3 | -14.266% |
| 4.0 | 0.5000 | 4.9117 | 0.566% | 8/1/3 | -26.284% |

The best uniform hindsight scale is `0.25`, but it regresses on the printed
pizza and pyramid. The sealed `0.125` scale is the largest member of the tested
uniform bank with no object regression. The prospective choice was therefore
conservative for a real reason, rather than merely leaving a uniformly better
setting unused.

## Remaining headroom

Selecting one multiplier per take in hindsight yields 4.8215 mm, a 2.392%
improvement over the checkpoint. Selecting one multiplier per frame yields a
2.956% improvement. These are capacity ceilings, not deployable results.

The preferred magnitude is strongly heterogeneous. The pizza and pyramid prefer
half of the sealed correction, while several plush and foam objects continue to
improve at three or four times the sealed correction. Support fraction alone
does not identify the two aggressive-scale failures. On the overlapping source
object, the pyramid was already the scale-fragile case: increasing the source
scale from `0.125` to `0.25` changed its gain from +0.214% to -1.979%. This is
useful development evidence for instance-conditioned shrinkage, but the overlap
is too small to validate a selector.

## Next method

The next prospective candidate should retain the sealed correction direction
and estimate only its scalar magnitude. A credible implementation would:

1. use earlier source interactions of the same physical object to form a
   hierarchical prior over the scale;
2. admit target-prefix adaptation only from outcome-independent association,
   action, correction-magnitude, and uncertainty summaries;
3. calibrate an upper bound on regret relative to scale `0.125` using source
   actions;
4. use scale `0.125` byte-for-byte whenever the adaptive gate is not certified;
5. freeze the rule before evaluating another untouched take cohort.

This is aligned with an instance-specific digital twin: the gain may be learned
from previous actions of the same object, while the next action remains held
out. A selector fitted from the fresh12 outcomes themselves would not be valid.

## Provenance

The diagnostic implementation is commit
`9e451d5d8eb8c7b0ef6580dce95f7bb99934bbf0`. It ran in a separate checkout on
`gpuserver4090` against the immutable fresh12 campaign. The audit has canonical
digest `78179996296b5ed47692e3ee716308c4525deeb71ce2881442331b5643b4bf94`
and file SHA-256
`114fbf7c3437311f625c9c022742a2c707d6db1e61ca1548b0d8f2500f83d494`.
