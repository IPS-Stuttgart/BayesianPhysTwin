# PokeFlex missing-five scale source result V5

## Result

The frozen 30-action source bank passed. It promotes a non-global discrepancy
scale for two of the five objects and returns the other three exactly to the
global `0.125` scale.

| Object | Source actions | Full/deployed multiplier | Mean gain vs global | Worst source gain | Minimum candidate LOO gain | Strict LOO wins | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `3dPrintedCylinder` | 6 | `2.0 / 2.0` | +1.992% | +1.130% | +0.999% | 6/6 | promote |
| `3dPrintedHeart` | 6 | `1.5 / 1.5` | +0.469% | +0.195% | +0.195% | 6/6 | promote |
| `3dPrintedPizza` | 6 | `1.0 / 1.0` | 0.000% | 0.000% | 0.000% | 0/6 | exact fallback |
| `Pillow` | 7 | `1.0 / 1.0` | 0.000% | 0.000% | -0.023% | 0/7 | exact fallback |
| `Sponge` | 5 | `1.0 / 1.0` | 0.000% | 0.000% | -0.489% | 0/5 | exact fallback |

Across all 30 deployed source actions, the mean relative improvement over the
global scale is **0.492%**. There are zero deployed source-action regressions
and zero deployed LOO regressions. Two diagnostic LOO candidate regressions,
for Pillow and Sponge, are retained in the artifact and trigger the intended
fallback. Synthetic controls detect 12/12 positive mechanisms and admit 0/12
placebos.

The frozen five-target candidate is therefore:

| Unavailable official target | Effective scale |
| --- | ---: |
| `3dPrintedCylinder_T7` | `0.25` |
| `3dPrintedHeart_T14` | `0.1875` |
| `3dPrintedPizza_T13` | `0.125` |
| `Pillow_T8` | `0.125` |
| `Sponge_T10` | `0.125` |

## Evidence and verification

- Implementation revision: `f461ec19e0032097bbdb97a1cbf5f4e01c3fde33`
- Protocol SHA-256: `83737068dca8621e331bcd30c76bc2852509872e59d034d984dc931d7bf5e27a`
- Protocol file SHA-256: `0671df8beaaa4e560a264599ab5edbedd2e66ad2a7e1f9181f1b71fdea5fc70a`
- Result canonical SHA-256: `49658508e9531abd43d966c0eeb56f4deec43db3234e0ea530f756955b6deee7`
- Result file SHA-256: `2f666ce4060a488f036745ff9471acd39a79e2c2a7e0799c7c03e65075e75bf1`
- V5 completion protocol SHA-256: `11d6eb1ff115f0021e1ab9ad959b0dfd614ca455e5f54d1dd05c99e9b916c7de`
- V5 completion protocol file SHA-256: `960eb903634c621b0ea2244a2039cb92da974f9a196d54c77622cdd40f2ab271`
- Complete source executions: 30/30
- Independent aggregate regeneration: byte-identical
- Raw artifact root: `/home/florianpfaff/source-only/pokeflex-missing5-scale-v5-run-83737068`

Repository verification before locking comprised 2,420 passing tests and 32
expected skips. One release-metadata test initially read stale metadata from a
shared virtual environment; it passed 5/5 against a fresh install of the same
checkout. The focused missing-five suite passes 9/9 and Ruff is clean.

## Claim boundary

All 30 calibration outcomes were public and previously exposed. The source
result establishes only cross-action stability and a pre-target scale choice.
It is not prospective target evidence and is not a state-of-the-art result.
No unavailable official target archive or outcome was read, the V4 official-18
protocol was not altered, and no held-v8 artifact was accessed.

## Recommendation

Freeze the table above as a separate V5 completion candidate and wait for the
five exact author-provided archives. Their evaluation is the independent,
predeclared test. Advance beyond V4 only if that target evaluation improves the
official aggregate without a material per-object regression. Do not tune these
five multipliers again from public actions or substitute alternate takes.
