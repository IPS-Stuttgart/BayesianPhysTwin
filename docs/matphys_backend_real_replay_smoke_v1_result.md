# Real MatPhys backend replay smoke v1

## Decision

The real MatPhys backend path passes its development smoke. On the already-open
`double_lift_zebra` case, an object-disjoint MatPhys `spring_Y` field was replayed
by official PhysTwin/Warp and exported as the same six-array physical contract as
the incumbent. The guarded selector accepted the candidate.

This result does not establish independent transfer or state of the art. It
justifies a fresh public-object source panel.

## Frozen boundary

- Bayesian-PhysTwin implementation: `a28bf33539026d03d9da933a7b3c65619c0c9bad`
- official PhysTwin: `2b6630528141b9cba5a7677c8b88b2129b4a8390`
- MatPhys: `c16b858dfb79bf21024ead24b45a710600de7b4f`
- MatPhys target-object evidence ended before frame 30
- selection interval: frames `[30,40)`
- future boundary: frame 40
- no future metric input was passed to either replay
- collision parameters and all non-spring physical parameters were unchanged

The MatPhys field was trained without the zebra object. The replay used the
known future actuator trajectory because action is part of the prediction
condition; the held-controller control kept all controllers at frame zero while
preserving gravity, support, collision, and material dynamics.

## Result

| Check | Incumbent | MatPhys/Warp | Change |
| --- | ---: | ---: | ---: |
| Validation CD | 5.054 mm | 4.833 mm | -4.37% |
| Validation track error | 7.867 mm | 5.720 mm | -27.29% |
| Balanced validation score | 1.000 | 0.842 | -15.83% |
| Validation trajectory difference | - | 3.062 mm RMSE | substantive |
| Identity replay RMSE | - | 0.000 mm | pass |

The candidate and incumbent share exact frame zero, persistence, action support,
shape, dtype, and node order. The selected backend archive has SHA-256
`98caca506ca3b7422a8a642d096ccbc885e6a6580827466171793d597ca31555`
and is a byte-exact copy of the candidate replay. The content-addressed backend
artifact ID is
`ef65cc79adca707fe7020336de0442f07cc2fc9bc0f50ff8b79540fadd48e3f0`.

The candidate spring geometric mean was 15.670 kPa-equivalent simulator units,
versus 30.056 for the incumbent field. This is a simulator spring-field change,
not a claim that either value is a directly identified material constant.

## Interpretation

The smoke closes the implementation gap: Bayesian-PhysTwin can now evaluate a
real MatPhys proposal as an alternate guarded physical backend, with an
independent identity replay and exact fallback. It also shows useful prefix
signal on one case. It does not answer whether that signal transfers across new
objects, remains calibrated, or improves the future benchmark mean.

The next protocol must therefore use genuinely fresh public objects, keep the
MatPhys training objects disjoint, freeze selection before future opening, and
require improvements in both primary metrics without unacceptable late-horizon
or worst-case regressions. Only a passing fresh source panel should authorize an
independent SOTA evaluation.
