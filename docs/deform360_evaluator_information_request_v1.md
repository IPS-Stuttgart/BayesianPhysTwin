# Deform360 evaluator information request

The public Deform360 repository now provides a complete annotation pipeline
and a PhysTwin interchange stage, but it explicitly does not release the
world-model baselines or benchmark evaluator. To reproduce the paper's
per-episode 3D table without guessing, please provide either the evaluator
source or a content-hashed contract resolving:

1. exact training object/episode membership;
2. exact evaluation object/episode membership and ordering;
3. whether the released control-point 80/20 split, persistent identities,
   world-frame metre coordinates, preprocessing, and all-true masks are also
   the table-evaluator contract;
4. exact Chamfer direction, distance power, frame reduction, and episode
   reduction;
5. exact track correspondence, distance, frame reduction, and episode
   reduction;
6. aggregation from episodes to objects and from objects to the reported
   total;
7. policy for failed, missing, invalid, or unequal-length predictions.

The request is limited to evaluation parity. It does not ask for unreleased
training code or model checkpoints.

Until these details are authoritative, Bayesian-PhysTwin reports explicit
candidate metric conventions and does not compare its local numbers directly
against the published table as an official state-of-the-art claim.
