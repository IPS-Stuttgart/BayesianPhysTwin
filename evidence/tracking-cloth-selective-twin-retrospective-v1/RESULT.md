# Tracking Cloth selective digital-twin result v1

## Status

**Bounded positive selectivity with a preregistered coverage shortfall.**

The registered aggregate decision is `retrospective-mixed-or-negative` because
the primary query/horizon gate accepted **10.00%** of query cases, below the
frozen **20.00%** minimum. Every other registered criterion passed.

This is not a wholly negative result. The frozen policy selected a narrow
regime in which the reduced physics model improved over persistence across all
four held-out materials, while exact fallback protected the unsupported
regimes.

## Execution and custody

- workflow run: `33518742754`
- scientific job: `99892372651`
- execution revision: `46c67d78ce98de3a67bc0751c22dc0c36a83b658`
- frozen scientific source: `e9fd0fc35c534db867831c124b46e495e5eb3e57`
- artifact ID: `9804820476`
- artifact SHA-256:
  `347d113c03768bbab6de3ca13cf1ee6a09e1cb2d1829e05c89d2a2f2faa90881`
- result ID:
  `25ab847afd8e4a90eef19c2a48cd4c257fe6bf88f6159f617ed5119fec7fa26f`
- authoritative `result.json` SHA-256:
  `4964ce1818046db979dc844d767038a07741306c99dfe2011b5508091e53a380`

The hosted run downloaded the official Zenodo archive and reproduced the
previously verified archive identities exactly:

- byte count: `27,377,271`
- MD5: `b4868b702f8a42b2ea1069d0f1a3b8f6`
- SHA-256:
  `14916efa89a26d991c024024cc9449397d3a6f654311e621bb91e9602e231e1a`
- dataset inventory ID:
  `b80986ea8af92b47bc8d3737eff5f2c796f292853746836e04ae1f8727feca91`

## Main result

| Policy | Coverage | Selected minus persistence | Practical harm among accepted | Material-bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Always persistence | 0.00% | 0.0000 mm | 0.00% | [0.0000, 0.0000] mm |
| Always physics candidate | 100.00% | +26.0142 mm | 56.17% | [+24.2415, +28.2960] mm |
| Motion-only gate | 0.00% | 0.0000 mm | 0.00% | [0.0000, 0.0000] mm |
| **Query/horizon gate** | **10.00%** | **-2.3952 mm** | **3.12%** | **[-3.1056, -1.6848] mm** |
| Outcome oracle diagnostic | 14.77% | -3.6260 mm | 0.00% | [-4.6199, -2.6321] mm |

Negative regret favors the selective policy.

The equal-material mean loss changed from **29.6625 mm** under persistence to
**27.2672 mm** under the selective policy, an **8.07% reduction**. Every held-out
material improved:

| Held-out material | Coverage | Selected minus persistence |
| --- | ---: | ---: |
| Cotton | 10.00% | -1.8801 mm |
| Denim | 10.00% | -3.0647 mm |
| Polyester | 10.00% | -3.1466 mm |
| Wool | 10.00% | -1.4895 mm |

The gate accepted 128 of 1,280 query cases. Of these, 117 were genuinely better
than persistence, 11 were worse, and four exceeded the practical harm margin.
This corresponds to:

- 91.41% beneficial precision among accepted cases;
- 61.90% recall of all candidate-beneficial cases;
- 67.72% of the oracle's accepted-case count; and
- 66.06% of the oracle's aggregate value.

Relative to always using physics, practical harm fell from **56.17%** to
**3.12%**, a 53.05 percentage-point or 94.44% relative reduction.

## Selected physical regimes

The frozen gate accepted exactly four context cells, all involving shaking and
long-horizon global free-marker queries:

| Motion | Query | Horizon | Coverage | Candidate minus persistence | Practical harm |
| --- | --- | ---: | ---: | ---: | ---: |
| Shake | Free-marker centroid | 2 s | 100% | -23.4295 mm | 6.25% |
| Shake | Free-marker centroid | 5 s | 100% | -27.9704 mm | 0.00% |
| Shake | Free-marker shape | 2 s | 100% | -20.6450 mm | 6.25% |
| Shake | Free-marker shape | 5 s | 100% | -23.7643 mm | 0.00% |

No twist query, short-horizon query, bottom-edge-centroid query, or
shape-radius query was admitted.

On the accepted cells, the Bayesian physics candidate beat persistence by
23.9523 mm on average and beat the MAP physics member by 2.6252 mm. It was,
however, **1.1862 mm worse than `last_residual` on average**. The present
experiment therefore does not establish superiority over the strongest
equally informed deterministic residual comparator.

## Registered criteria

- minimum selected coverage of 20%: **fail** (`10%`)
- negative equal-material regret: **pass**
- negative material-bootstrap upper 95% bound: **pass**
- nonpositive mean regret for every held-out material: **pass**
- zero exact-fallback violations: **pass**
- lower practical-harm rate than always-candidate physics: **pass**

The frozen 20% support requirement may be ambitious relative to the observed
outcome-oracle coverage of 14.77%, but it cannot be changed after opening the
outcomes. The registered decision must therefore remain mixed/negative.

## Scientific interpretation

The result supports the bounded mechanism claim that **query and forecast
horizon identify a narrow physical regime in which a reduced simulator is
useful**, and that exact fallback prevents the severe errors produced by
always using the simulator.

It does not support broad simulator competence. In particular:

- the simulator is harmful when used unconditionally;
- the accepted regime covers only 10% of the registered task portfolio;
- no twist context is supported;
- the Bayesian candidate does not beat `last_residual` on the accepted cells;
- the data had prior target exposure and this is not fresh confirmation.

## Claim boundary

This is retrospective cross-material feasibility evidence on the already-open
public Tracking Cloth free-hanging factorial. It cannot establish fresh
physical confirmation, deployment safety, calibrated joint uncertainty,
unseen-action transfer, universal simulator validity, or state of the art.
