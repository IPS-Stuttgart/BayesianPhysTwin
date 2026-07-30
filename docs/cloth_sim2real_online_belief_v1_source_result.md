# Cloth Sim2Real v1 source result

## Evidence boundary

The guarded readout method was frozen at commit `3c8f34e` after one declared
development smoke (`chequered_rag_0/dynamic`). The remaining five repeat-0
cases were then run without code or hyperparameter changes. Every prediction
artifact was sealed before its future point clouds were opened.

No repeat-1 calibration point cloud and no repeat-2 target point cloud was
opened for this result. The compact source-gate artifact has SHA-256
`51d33e4753285d1c5124e040c12b505d3b52ffa41ed80a761f45788e03b6217e`.

## Primary result

The frozen primary source gate passed:

| Source dynamic case | Physical symmetric L1 CD | Guarded CD | Relative change |
| --- | ---: | ---: | ---: |
| Chequered rag | 68.80 mm | 62.21 mm | -9.58% |
| Cotton rag | 102.34 mm | 91.86 mm | -10.24% |
| Linen rag | 99.65 mm | 97.84 mm | -1.82% |
| Object-balanced | | | **-7.21%** |

All three cloths improved, satisfying the preregistered 5% object-balanced
dynamic improvement and three-of-three non-regression gate. The released
simulator-to-observation directed L1 companion metric improved by 9.05%,
5.97%, and 1.37% over the full causal future, respectively.

## Secondary result and uncertainty

The quasi-static task did not pass as a positive transfer result:

| Source quasi-static case | Physical symmetric L1 CD | Guarded CD | Relative change |
| --- | ---: | ---: | ---: |
| Chequered rag | 39.46 mm | 38.95 mm | -1.30% |
| Cotton rag | 58.72 mm | 62.43 mm | +6.32% |
| Linen rag | 61.99 mm | 60.73 mm | -2.04% |
| Object-balanced | | | **+0.99%** |

Raw nominal 90% coordinate coverage was 37.7--56.4% on the three dynamic
cases and 67.8--86.8% on the quasi-static cases. The source result therefore
supports the dynamic conditional mean update, but not a calibrated Bayesian
claim. The cotton quasi-static regression also shows that a strong prefix gate
does not guarantee long-horizon transfer under contact-rich motion.

## Decision

The source gate authorizes repeat-1 calibration under the frozen method. It
does not authorize repeat-2 target evaluation. Calibration must independently
meet:

1. at least 5% object-balanced dynamic improvement;
2. at least two of three dynamic cloth wins;
3. no dynamic cloth regression greater than 5%;
4. uncertainty calibration fitted only from repeat-1 outcomes.

Failure of any gate keeps repeat 2 sealed. Even if calibration passes, the
eventual claim is causal online continuation from a real prefix, not
identical-information open-loop superiority over the original benchmark.
