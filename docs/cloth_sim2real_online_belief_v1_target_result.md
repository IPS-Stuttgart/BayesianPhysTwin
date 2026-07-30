# Cloth Sim2Real v1 target result

## Evidence boundary

Repeat 2 was opened only after the source and calibration accuracy gates
passed. The method, candidate bank, prefix boundaries, admission rule, and
uncertainty multiplier were frozen at commit `293c9ea`. All six target
predictions were sealed before their future point clouds were read, and each
result binds the prediction seal and the calibration-gate SHA-256
`332a278d3d47d4d5a05fa8723f1e4bb3b41c690ab88a381905f46cad79d0c2b8`.

The compact target artifact has SHA-256
`2b013b3e3214b4c8a1a6838b31fe304b6ea91df6b2e2610403ca37948feaa49a`.

## Dynamic primary result

The frozen guarded update improved symmetric L1 Chamfer distance on all three
independent target cloth trials:

| Target dynamic case | Physical symmetric L1 CD | Guarded CD | Relative improvement |
| --- | ---: | ---: | ---: |
| Chequered rag | 73.05 mm | 65.77 mm | 9.97% |
| Cotton rag | 96.12 mm | 85.84 mm | 10.70% |
| Linen rag | 85.55 mm | 84.06 mm | 1.75% |
| Object-balanced | 84.91 mm | 78.55 mm | **7.47%** |

The object-balanced directed simulator-to-observation metric improved 5.18%
over the full causal future and 4.11% over the benchmark's released dynamic
comparison windows. Symmetric L2 Hausdorff distance improved 9.43%.

The gain increased with forecast horizon:

| Future third | Physical symmetric L1 CD | Guarded CD | Relative improvement |
| --- | ---: | ---: | ---: |
| Early | 74.69 mm | 73.13 mm | 2.12% |
| Middle | 90.20 mm | 85.29 mm | 5.54% |
| Late | 90.16 mm | 77.40 mm | **13.60%** |

This confirms the preregistered positive domain: causal continuation of
dynamic cloth motion after a real observation prefix. The method is a
readout-belief update; it does not claim to correct the underlying MuJoCo
state or material parameters.

## Quasi-static secondary result

The same frozen method did not transfer to quasi-static motion:

| Target quasi-static case | Physical symmetric L1 CD | Guarded CD | Relative improvement |
| --- | ---: | ---: | ---: |
| Chequered rag | 48.07 mm | 56.96 mm | -18.48% |
| Cotton rag | 58.49 mm | 61.96 mm | -5.94% |
| Linen rag | 61.48 mm | 61.47 mm | 0.02% |
| Object-balanced | 56.01 mm | 60.13 mm | **-8.13%** |

The early quasi-static third improved 10.62%, but the middle and late thirds
regressed 9.88% and 33.97%. This is consistent with contact-rich
quasi-static evolution leaving the validation domain of a persistent prefix
correction. It is a locked negative result, not a target for post-hoc tuning.

## Uncertainty

Raw nominal 90% coordinate coverage was 46.21% on dynamic trials. Applying
the calibration-only standard-deviation multiplier of 5.05205 raised target
coverage to 93.65%, but the mean interval width was 168.61 mm. Quasi-static
coverage was 97.38% with a 250.79 mm mean width.

The scaled intervals are therefore conservative but not sharp. They are also
not a formal 90% split-conformal result: six calibration trials do not provide
the required finite-sample order-statistic resolution. Future work should
improve the covariance model rather than interpreting broad temperature
inflation as calibrated Bayesian uncertainty.

## Claim

Across one development repeat, one independent calibration repeat, and one
independent target repeat, the frozen guarded belief update consistently
improved dynamic continuation:

| Split | Object-balanced dynamic improvement | Cloth wins |
| --- | ---: | ---: |
| Source | 7.21% | 3/3 |
| Calibration | 6.45% | 2/3 plus one exact fallback |
| Target | **7.47%** | **3/3** |

This is independent target evidence that a prefix-conditioned,
baseline-relative guarded observation update improves the released physical
rollout for dynamic cloth continuation. It is not an identical-information
open-loop state-of-the-art comparison: the method observes a real prefix,
whereas the original benchmark evaluates physical rollouts under a different
information contract. Quasi-static motion and sharp uncertainty calibration
remain unresolved.
