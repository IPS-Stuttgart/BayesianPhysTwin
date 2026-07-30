# Cloth Sim2Real multi-backbone exploratory v2 result

## Result

The official SOFA v23.06 binary produced finite, topology-consistent rollouts
for all three dynamic source cloths. Applying the frozen v1 guarded readout
update improved two cloths and rejected the third with an exact physical
fallback:

| Opened source case | SOFA directed CD | Guarded directed CD | Gain | Decision |
| --- | ---: | ---: | ---: | --- |
| Chequered rag | 91.63 mm | 76.50 mm | 16.52% | admitted |
| Cotton rag | 93.88 mm | 76.91 mm | 18.08% | admitted |
| Linen rag | 91.04 mm | 91.04 mm | 0.00% | exact fallback |
| Object-balanced | 92.19 mm | 81.48 mm | **11.61%** | 2/3 admitted |

The object-balanced symmetric L1 Chamfer improvement was 9.40%. On the
benchmark's released dynamic comparison windows, directed CD improved from
94.67 mm to 81.92 mm, or 13.46%. Late-horizon symmetric CD improved 8.14%.

This is positive evidence for **backbone compositionality**: the update was
frozen on MuJoCo and still helped two independently generated SOFA rollouts.
It is not evidence that this SOFA execution is state of the art.

## Parity failure

The benchmark reports mean dynamic SOFA directed CD of approximately 67 mm,
75 mm, and 61 mm for chequered, cotton, and linen cloth. The reproduced
source-repeat physical values over the corresponding released windows were
99.64 mm, 93.10 mm, and 91.26 mm.

The runtime used:

- benchmark commit `178a9b9722191c51cf0dcbc3cf0dc03701b09eb3`;
- official SOFA v23.06 archive SHA-256
  `de1ab962978f1b77db97d9925e6fef6b2bc924aff6aa04956a59d9e1bd0e3720`;
- SOFA commit `c58927d2920fb7a1b0826c462d9c02bb2f0fa819`;
- adapter commit `c6558b5`, file SHA-256
  `a040b703f0026d6502eed0ceba188ab57bf417dfdda39cf50fe5ec8a818cd8a7`.

The official binary may differ from the authors' source-built Docker runtime,
and the published table aggregates repeated seeds. Until the authors'
container or released simulator trajectories reproduce the reference metric,
the physical-prior difference is a runtime-parity issue rather than a method
comparison.

## Gate decision

The preregistered exploratory advancement rule required guarded improvement
on all three dynamic cloths and a physical baseline in the published regime.
Both conditions failed:

- guarded improvement occurred on 2/3 cloths;
- linen used exact fallback;
- paper-level SOFA parity was not reproduced.

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
