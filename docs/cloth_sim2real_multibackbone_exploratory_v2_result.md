# Cloth Sim2Real multi-backbone exploratory v2 result

## Result

The official SOFA v23.06 binary produced finite, topology-consistent rollouts
for all three dynamic source cloths. Applying the frozen v1 guarded readout
update improved two cloths and rejected the third with an exact physical
fallback:

| Opened source case | SOFA directed CD | Guarded directed CD | Gain | Decision |
| --- | ---: | ---: | ---: | --- |
| Chequered rag | 82.67 mm | 70.40 mm | 14.85% | admitted |
| Cotton rag | 115.48 mm | 95.50 mm | 17.30% | admitted |
| Linen rag | 85.73 mm | 85.73 mm | 0.00% | exact fallback |
| Object-balanced | 94.63 mm | 83.88 mm | **11.36%** | 2/3 admitted |

The object-balanced symmetric L1 Chamfer improvement was 7.55%. On the
benchmark's released dynamic comparison windows, directed CD improved from
104.53 mm to 89.87 mm, or 14.03%. Late-horizon symmetric CD improved 4.29%.

This is positive evidence for **backbone compositionality**: the update was
frozen on MuJoCo and still helped two independently generated SOFA rollouts.
It is not evidence that this SOFA execution is state of the art.

## Corrected parity audit

The initial adapter incorrectly reused MuJoCo's one-second settling interval
for SOFA. The official benchmark runs ten seconds of SOFA settling. After
matching that backend-specific contract, the benchmark's full pre-contact
window was scored over all three real repeats:

| Cloth | Reproduced SOFA CD | Published SOFA CD | Absolute difference / published SD |
| --- | ---: | ---: | ---: |
| Chequered | 76.63 mm | 68 +/- 24 mm | 0.36 |
| Cotton/towel | 111.61 mm | 78 +/- 29 mm | 1.16 |
| Linen | 66.11 mm | 61 +/- 24 mm | 0.21 |

The earlier note also compared the post-prefix online continuation window
against the paper's full pre-contact benchmark window and transposed the
published towel and chequered values. Those comparisons were invalid. The
corrected single-rollout reproduction does not equal the paper's 20-seed
means, but every cloth lies within approximately 1.2 reported standard
deviations. The physical baseline is therefore in the published regime,
although exact runtime parity is not established.

The runtime used:

- benchmark commit `178a9b9722191c51cf0dcbc3cf0dc03701b09eb3`;
- official SOFA v23.06 archive SHA-256
  `de1ab962978f1b77db97d9925e6fef6b2bc924aff6aa04956a59d9e1bd0e3720`;
- SOFA commit `c58927d2920fb7a1b0826c462d9c02bb2f0fa819`;
- adapter commit `b7e684c`, file SHA-256
  `da8ffe782d084da31aa36f3a700624fcaed43ce10aaa7f0cf2a63f20d0bcfded`.

The official binary may still differ from the authors' source-built runtime,
and the published table aggregates 20 random seeds and three real repeats.
That residual uncertainty remains part of the claim boundary.

## Gate decision

The preregistered exploratory advancement rule required guarded improvement
on all three dynamic cloths and a physical baseline in the published regime.
The method condition failed:

- guarded improvement occurred on 2/3 cloths;
- linen used exact fallback;
- object-balanced and late-horizon metrics remained positive;
- the corrected physical baseline was in the paper's reported regime.

Therefore this branch does **not** authorize a larger preregistered
evaluation. The useful result is narrower: guarded Bayesian readout updates
can transfer across simulator families without regression, provided an exact
fallback is retained.

## Next action

Do not tune the update on these opened repeats. Revisit this route only if one
of the following becomes available:

1. the exact authors' SOFA container;
2. released SOFA trajectories with provenance;
3. a fresh cloth dataset on which simulator choice and update gates can be
   locked before outcomes.

The primary SOTA effort remains the independently frozen guarded online-belief
evaluation; this exploratory Cloth Sim2Real branch is a supporting
cross-simulator ablation.
