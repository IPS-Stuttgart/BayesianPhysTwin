# Guard-aware uncertainty: frozen development result

## Decision

**Primary advancement gate FAIL. Positive log-score signal, but no promotion.**
This study keeps the previous eight-observation paired physical point forecast
byte-identical. The new fixed-mean uncertainty surrogate improves aggregate NLL,
but fails the registered per-object precision and volume requirements. No fresh
evaluation or protected target is authorized by this result.

One frozen CPU prediction invocation completed all 30 trajectories in 0.984 s,
reusing verified physical responses and prefix posteriors. There were no new
native/GPU rollouts, queries, recordings, replacements, technical failures, or
unsealable trajectories. This time excludes the already-completed parent model
and response generation, calibration, scoring, and verification. All predictions
were sealed before calibration; DLO2-only calibration was sealed before opened
DLO1/DLO3 transfer scoring. The write-once attempt ledger was consumed once.

## Main results

These are equal-object DLO1/DLO3 results over 16 already-open trajectories, with
source-only calibration on 13 DLO2 trajectories. Every row shares exactly the
same point forecast: coordinate L1 8.711 mm, hidden point RMSE 20.572 mm. NLL is
in nats per 3D marginal event with coordinates in metres; lower is better.
Width is the geometric-mean full axis diameter of the 90% ellipsoid.

| Covariance, all with primary moment calibration | NLL | 90% coverage | Width (mm) | Mean volume (mm^3) |
|---|---:|---:|---:|---:|
| Isotropic source calibration | -8.839 | 91.08% | 69.31 | 192,663 |
| Mean-guard-scaled tangent covariance | -8.850 | 91.38% | 69.67 | 194,892 |
| Unguarded shadow covariance | -8.885 | 91.55% | 69.31 | 192,746 |
| Source-fitted full 3x3 second moment | -8.893 | 90.13% | **67.18** | **174,327** |
| Fixed-mean bridge, primary | **-9.028** | 91.71% | 67.23 | 183,750 |
| Cyclic-axis-rotated bridge | -8.723 | 91.67% | 72.39 | 225,547 |

Relative to isotropic uncertainty the primary improves NLL by 0.188 nats,
reduces aggregate width by 3.01%, and reduces aggregate volume by 4.63%. Against
the stronger source-full comparator it improves aggregate NLL by 0.135 nats but
increases volume by 5.41%, with essentially unchanged width (+0.068%). These are
descriptive opened-data comparisons, not a passed transfer gate.

The rotation control preserves raw eigenvalues/volume but is worse after the
same source-only calibration procedure. This is a useful directional diagnostic,
not independent confirmation that the physical shadow posterior is correct.

## Why the frozen gate failed

| Object | Primary NLL | NLL wins vs isotropic / source-full | Volume change vs isotropic / source-full | Gate |
|---|---:|---:|---:|---|
| DLO1 | -8.857 | 8/8 / 6/8 | +7.56% / +18.88% | FAIL: larger volume |
| DLO3 | -9.198 | 7/8 / 7/8 | -16.81% / -8.07% | FAIL: NLL intervals include zero |

Whole-trajectory bootstrap 95% intervals for primary-minus-comparator NLL:

| Object | Isotropic | Source-full | Unguarded shadow |
|---|---|---|---|
| DLO1 | [-0.5505, -0.0805] | [-0.4436, -0.0312] | [-0.3842, -0.0613] |
| DLO3 | [-0.2188, +0.0123] | [-0.1895, +0.1062] | [-0.1922, +0.0385] |

Both objects meet the coverage gate (91.15% and 92.27%). The declared composite
gate nevertheless requires every condition on each object, not just favorable
aggregate means or many trajectory wins. It cannot be weakened after these
results. With only two transfer objects, these intervals do not estimate a
population of unseen physical objects.

## Calibration is still limited

Moment calibration gives mean source NEES exactly three by construction, but
primary DLO2 coverage is only 86.01%. It is not a nominal-coverage guarantee.
The fixed secondary conformal procedure is also reported, not selected on
transfer outcomes:

| Covariance with secondary conformal calibration | NLL | Coverage | Width (mm) |
|---|---:|---:|---:|
| Isotropic | -7.840 | 98.58% | 134.24 |
| Source-full | -7.897 | 98.71% | 131.47 |
| Fixed-mean bridge | -7.970 | 99.02% | 129.56 |

Rank 13/13 uses the maximum source-trajectory score and is necessarily coarse.
The resulting wide, overcovering transfer ellipsoids do not rescue the primary.
These are marginal 3D scores, not a joint distribution over all coordinates or
a simultaneous trajectory coverage guarantee.

## What this contributes

The useful reporting distinction survives the empirical gate failure: capping
or rejecting a deployed mean is not evidence that physical uncertainty vanished.
For shadow mean `mu`, shadow covariance `C`, and a fixed deployed forecast `a`,
the Gaussian log-score-optimal second moment is `C + (mu-a)(mu-a)^T` under the
assumed shadow distribution. This identity is standard mathematics, not a new
theorem or proof that an approximate physical belief is calibrated.

The experiment adds a tested, baseline-preserving implementation and strong
controls. It supplies a tentative signal that physically aligned uncertainty can
help a fixed point predictor, but **not** a validated new uncertainty method,
point-accuracy gain, official benchmark result, or SOTA claim. No existing method
or frozen result was replaced.

The stronger point-evidence result remains the previous paired update. The
separate matched weak-constraint study measured 20.572 mm hidden RMSE for that
eight-query update, versus 21.512 mm for sixteen-query OLS plus physical replay
and 21.914 mm for a sixteen-query DEFORM-style periodic position control. That
comparison strengthens the experimental story without claiming periodic state
correction itself is new or reproducing the official DEFORM camera pipeline.

Recommendation: retain the existing point method; archive this as an exploratory
UQ lead and the weak-constraint/forecast-sensing extensions as failed primary
gates. Do not continue selecting caps, floors, or calibrators on these outcomes,
and do not proceed to a larger target study under a failed advancement rule.
Any future advancement needs a distinct source-validated protocol. Public data
can support that program; new physical recordings were not used here.

## Verification and immutable provenance

- Frozen implementation, protocol, tests, and checker:
  `100fcf5c5c8a8d98054601d8372b8aa795646cb6`.
- 453 relevant local tests; 33 focused tests in the exact remote runtime;
  Ruff and focused MyPy pass. The full repository suite was not run.
- All 960 frozen source files verified. All 30 deployed means are byte-identical.
- Independent matrix multiplication, source calibration, Cholesky scores,
  aggregate metrics, bootstrap intervals, and gate recomputation pass.
- 150 covariance carriers, 348 trajectory/arm/calibration records, and 167,040
  marginal UQ events verified. Local delivery: three NPZs and 24 arrays verified.
- No empirical or verifier amendment/retry. No old result or incumbent source edit.

| Artifact | SHA-256 |
|---|---|
| Protocol | `f6ef402c86749729325ff6e1bbe25d926cae4e617caf041baab4b5ee4031b2ef` |
| Source receipt | `7353b787735e6dde8cf83d474cc5112b4e7ed3a6490027b1227af7bf9c5acfed` |
| Prediction barrier | `a8a11744efd884735928db5230d96f4dbb830a76c97163b4c3aa0e5ecfbce682` |
| Calibration | `450ecb0edefbf1672598d0a145b30c9d388a4c094397e28e0eb306414988d24f` |
| Result | `12810450b05c43bc0afa78ed134a8ad1215ccdf0ebfa197fdeee166fbbf0f661` |
| Independent verification | `5fffffae8af8048ca969c4a59ce533739103028f29a2aca1e4672e480b3924cc` |

Full evidence remains in the source-only server run and local user archive.
This result is local/private-paper evidence only; it was not pushed or merged.
