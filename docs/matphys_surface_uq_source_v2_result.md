# Guarded MatPhys surface-UQ source result v2

## Decision

The frozen source gate failed. Three of ten registered source cases were
scorable, below the preregistered minimum of eight. No fresh target evaluation
is authorized, and the frozen DEFORM mean remains byte-identical and unchanged.

This is an endpoint-qualification failure, not a measured MatPhys NLL or
coverage loss. The protocol correctly did not compute those metrics on an
under-supported subset.

## Accounting

| Case | Policy | Surface support | Frozen disposition |
| --- | --- | ---: | --- |
| `026-sock-cloth-ep0007` | MatPhys | 48.07% | insufficient support |
| `031-cotton-cloth-ep0000` | MatPhys | 4.38% | insufficient support |
| `036-napkin-cloth-ep0009` | MatPhys | 15.08% | insufficient support |
| `058-roll-napkin-ep0001` | none | unavailable | unavailable physical carrier |
| `152-slime-ep0008` | MatPhys | 45.57% | insufficient support |
| `153-cake-ep0005` | MatPhys | 96.89% | scorable |
| `167-glove-gray-cloth-ep0000` | none | unavailable | unavailable physical carrier |
| `186-monster-ep0006` | exact isotropic fallback | 99.84% | scorable |
| `193-frog-ep0007` | MatPhys | 88.28% | scorable |
| `198-kneepad-cloth-ep0002` | MatPhys | unavailable | retained technical failure |

All eight reconstructable endpoints completed with all 12 registered disjoint
scoring cameras. The `198` scorer then failed closed because the DEFORM mean
contains 1,069 graph nodes while the MatPhys covariance contains 1,028. The
result retains the case as a technical failure rather than truncating,
reindexing, or dropping identities after opening the source endpoint.

The `186` case exercises the other intended guard branch: its sealed MatPhys
signal-to-effective-replay-floor ratio was 1.284, below the fixed threshold of
2, so it received the exact leave-one-case-out isotropic comparator covariance.
That fallback is a tie rather than a MatPhys win.

## Frozen evidence

- Protocol: `configs/sota/matphys_surface_uq_source_v2.json`
- Source result: `results/sota/matphys_surface_uq_source_v2/source_result.json`
- Result ID: `7a86919a989e8ae6aca5f0e85cf6667350f81f1ff53432c0a8f84448e8c11a5e`
- Result file SHA-256: `188abf20ec58452a060920e346e93b8ce12e9f301ecb8d3b6451b717149748a5`
- Durable run root: `/mnt/corsair/florianpfaff/matphys-fold-ensemble-transfer-v1`
- Frozen implementation before source outcome: `0a36bf5f1ee0bc7b455dca6025590617f7160893`

The result records `target_or_confirmation_data_read=false`,
`held_v8_artifacts_accessed=false`, `dlo4_or_dlo5_accessed=false`,
`fresh_target_authorized=false`, and `frozen_deform_results_changed=false`.

## Interpretation

The experiment validates useful infrastructure: independent official-Warp
parity, target-free uncertainty admission, exact fallback, complete-denominator
accounting, and byte-identical DEFORM means. It does not establish whether the
MatPhys ensemble covariance improves NLL, coverage, or decision safety because
the custom disjoint-camera surface endpoint did not qualify often enough.

The next MatPhys uncertainty evaluation needs an outcome contract with stable
material identity and graph alignment, or a preregistered deterministic mapping
between backend and scored identities. Lowering the support threshold or
silently truncating the `198` graph would change the estimand after observing
source outcomes and is therefore prohibited.
