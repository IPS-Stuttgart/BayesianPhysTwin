# Five-backend support matrix v1

**Status:** installed, source-verifiable contract  
**Scope:** DEFORM DLO, MatPhys/Warp, JAX-FEM v2, MuJoCo Flex, and SOFA FEM v3

## What fully supported means

`fully-supported` is an integration status, not a performance verdict. Every
backend in this matrix has all of the following:

1. a discoverable execution interface;
2. a typed artifact or frozen protocol;
3. deterministic validation;
4. retained positive or negative evidence;
5. exact incumbent fallback;
6. registered regression tests;
7. release-compatible packaging or an explicitly isolated source runner; and
8. a documented scientific claim boundary.

This definition protects working backends from experimental additions while
allowing a failed scientific gate to remain a first-class result. The separate
`recommendation authorized` column is the only user-facing predictive
promotion signal.

## Current matrix

| Backend | Execution surface | Highest retained decision | Fully supported | Recommendation authorized |
| --- | --- | --- | ---: | ---: |
| DEFORM DLO v7 | Isolated repository-source predictor and frozen DLO protocol | Official DLO2 benchmark value qualified | yes | yes, DLO2 only |
| MatPhys with official PhysTwin Warp | Installed adapters plus source runner | Native source covariance value rejected | yes | no |
| JAX-FEM finite deformation v2 | Installed adapter plus pinned external runtime | Source physics passed; source-value rollout physically rejected | yes | no |
| MuJoCo Flex | Installed adapter plus `mujoco-flex` optional runtime | Native smoke passed; source physics rejected | yes | no |
| SOFA FEM v3 | Installed adapter plus pinned external runtime | Source physics passed; source-value rollout physically rejected | yes | no |

DEFORM remains outside the generic material-solver registry because it is a
specialized differentiable-rod predictor with its own training and evaluation
contract. Its source runner is intentionally excluded from the stable wheel so
that PyTorch/upstream training dependencies cannot destabilize the portable
BayesianPhysTwin package. The protocol, tests, result receipts, and exact
fallback remain fully maintained in the repository.

MatPhys is also not a separate simulator family. It proposes spring/contact
parameters and replays them through the official PhysTwin Warp simulator. Its
backend and ensemble interfaces are installed; its retained source covariance
candidate did not pass promotion.

JAX-FEM, MuJoCo Flex, and SOFA use the existing simulator-neutral Lagrangian or
material-trajectory transports. Their native engine dependencies are loaded
only by explicit execution functions, so importing BayesianPhysTwin does not
import an optional simulator.

## Machine-readable API

The installed descriptor is available without any optional simulator:

```python
from bayesian_phystwin.backend_support_v1 import (
    describe_five_backend_support,
    verify_five_backend_source_tree,
)

support = describe_five_backend_support()
print([(item["backend_id"], item["evidence"]["stage"]) for item in support["backends"]])

# In a source checkout, also rehash every declared implementation, test,
# documentation, and retained evidence path.
print(verify_five_backend_source_tree("."))
```

The installed JSON resource is SHA-256 bound. Source verification fails closed
if an implementation/test/document path disappears, an evidence artifact
changes, a protected-target flag is set, or support is confused with predictive
promotion.

## Evidence boundaries

### DEFORM DLO v7

The frozen physical model plus causal local residual achieved `7.8606 mm` mean
coordinate L1 on all 14 unique released DLO2 trajectories, compared with
`8.7470 mm` for its identically trained physical checkpoint, with `14/14`
paired wins. The claim is restricted to that exact DLO2 contract.

### MatPhys/Warp

The native source covariance panel accounted for all 11 registered cases: 8
ordinary scores and 3 retained native-parity failures. The candidate had `0/8`
NLL wins and was rejected. This is maintained implementation support, not a
recommendation.

### JAX-FEM v2

The pinned finite-deformation runtime passed native smoke and two-group
source-physics qualification. Its frozen full-horizon source-value arm crossed
the hard orientation gate before source outcomes were read. Exact fallback is
the supported behavior.

### MuJoCo Flex

The native volumetric smoke passed. The registered source replay inverted at
substep 3 of 334, so source value was never authorized. Exact fallback is the
supported behavior.

### SOFA FEM v3

The pose-canonical stable-Neo-Hookean runtime passed native smoke and two-group
source-physics qualification. Its frozen source-value generation crossed the
hard determinant threshold before outcomes were read. Exact fallback is the
supported behavior.

No entry in this matrix opens held-v8, DLO4/DLO5, or another protected target.
The matrix records existing public/source evidence and does not authorize a new
empirical run.
