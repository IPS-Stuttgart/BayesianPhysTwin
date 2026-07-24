# Equivariant generalized-force v1 preflight

Status: rejected before Stage 1 on 2026-07-24.

The source episode builder completed all 17 released source cases without
opening a target artifact. That preflight exposed a unit-contract error:
PhysTwin's released graphs assign every node simulation mass `1.0`; those values
are not kilograms. V1 nevertheless labeled mass-scaled residual acceleration as
Newtons and imposed a fixed `0.5 N` cap.

The resulting target cap fraction ranged from 43.43% to 91.51% across cases.
No source model was trained and no official-Warp outcome was evaluated under
that contract.

V2 replaces the invalid SI label with native Warp generalized-force units and
uses a prefix-only robust case scale. The negative preflight remains useful
provenance: it prevented a numerically trainable but physically misleading
result from entering the evidence chain.
