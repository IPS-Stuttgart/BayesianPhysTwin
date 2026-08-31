# Query-conditional simulator competence certificates v2

This release extends the immutable v1 synthesis with the prospective
reward-aligned Slingshot v4 result. The v1 document and atlases v2-v4 remain
byte-identical historical evidence.

## Updated evidence

| Exact query | Gain over baseline | Paired 95% gain CI | Harmed worlds | One-sided 95% harm upper | Retained candidate gain | Oracle headroom | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DLO-Lab wrapping v9 | +0.004721 | [0.003894, 0.005597] | 1/288 | 0.016365 | 19.43% | 8.70% | **Certified** |
| DLO-Lab Slingshot v2 | +0.000220 | [-0.000111, 0.000530] | 14/288 | 0.074952 | 1.38% | 0.86% | **Rejected** |
| DLO-Lab Slingshot reward-aligned v4 | +0.003457 | [0.001514, 0.005711] | 6/288 | 0.040703 | 24.48% | 13.80% | **Certified** |

Slingshot v2 and v4 are not competing summaries of one unchanged experiment.
Their observation policies and statistical units differ, so atlas v5 assigns
different query IDs. V2 remains rejected. V4 is certified on a complete new
denominator.

V4 uses 128 fresh calibration worlds and 288 fresh evaluation worlds. It treats
rare late native-contact bifurcation as process variability while keeping
duplicate reward agreement within 0.001 as a hard gate. The incumbent reward
averages two independently executed baseline slots. All 3,328 one-action native
processes completed ordinarily, with no retry, replacement, or partial score.

The candidate decisions were sealed before any evaluation future. The guard
updated 36/288 worlds and improved reward by `+0.003457`, with paired 95%
interval `[0.001514, 0.005711]`. It reduced harmful worlds from 69 under the
unguarded posterior policy to 6 and retained a one-sided harm upper bound of
`0.040703`. It also beat the equal-data simultaneous-regret guard by
`+0.004338`, with paired interval `[0.001935, 0.006973]`.

The cross-query conclusion is stronger than a universal backend label:
exact-fallback Bayesian guards have prospective decision value on wrapping and
reward-aligned Slingshot, while rejected Slingshot v2 and three other public
simulator queries demonstrate that competence does not transfer by simulator
name alone.

## Immutable evidence

- Slingshot v4 compact result:
  `2882809b7265714a93be2d3f1455eeac527adbe681cc990cde762777fcaf3a85`.
- Query competence atlas v5:
  `82aef94511f3e0db1746262d4d49ae3ff9e52a587c5c11ce41cc817faa7a7ab9`.
- Atlas v5 file SHA-256:
  `2c9b13c10f6d89ca568bcdcd9fc1cc5b30d9443863323fac7478ccfe8541766a`.

Rebuild the non-pooling atlas with:

```bash
PYTHONPATH=src python scripts/build_dlolab_query_competence_atlas_v5.py \
  --output /tmp/dlolab-query-competence-atlas-v5.json
```

## Claim boundary

This is public-simulator evidence, not an official benchmark or SOTA claim,
distribution-free guarantee, independent human review, real-robot safety
certificate, or evidence for arbitrary unseen worlds. The v4 roster, method,
thresholds, and one-attempt result are closed. No new recordings, protected
targets, held-v8, DLO4, or DLO5 were used.
