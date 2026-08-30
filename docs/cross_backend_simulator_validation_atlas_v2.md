# Cross-Backend Simulator Validation Atlas v2

## Scope

Atlas v2 preserves every v1 entry and its exact decisions, then adds three
already-frozen native-continuum source qualifications from public PhysTwin
source interactions:

- MuJoCo Flex 3.9 volumetric mechanics;
- JAX-FEM v2 hyperelastic mechanics; and
- SOFA FEM v3 canonical-gauge mechanics.

No simulator was rerun. No source outcome, protected target, held-v8, DLO4, or
DLO5 artifact was opened. The three source records were already terminal and
retain the byte-exact incumbent fallback.

## Result

| Exact query | Runtime | Native | Full horizon | Headroom | Source | Prospective | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DLO-Lab wrapping | pass | pass | pass | pass | pass | pass | **certified** |
| DLO-Lab slingshot | pass | pass | pass | fail | fail | fail | rejected |
| DLO-Lab coiling | pass | pass | pass | fail | fail | -- | rejected |
| DLO-Lab separation | pass | fail | -- | -- | -- | -- | rejected |
| DLO-Lab unknotting | pass | fail | -- | -- | -- | -- | rejected |
| ARCSim Dirichlet | pass | pass | pass | n/a | fail | -- | rejected |
| Codim-IPC | pass | pass | fail | n/a | -- | -- | rejected |
| LibuIPC ensemble | pass | pass | fail | n/a | -- | -- | rejected |
| MatPhys pinned runtime | fail | -- | -- | n/a | -- | -- | rejected |
| MuJoCo Flex | pass | fail | -- | n/a | -- | -- | rejected |
| JAX-FEM v2 | pass | pass | fail | n/a | -- | -- | rejected |
| SOFA FEM v3 | pass | pass | fail | n/a | -- | -- | rejected |

Across 12 exact queries, 8 backend identities, and 3 public datasets:

- runtime execution passes in 11/12;
- native qualification passes in 8/12;
- complete-horizon qualification passes in 4/12 and fails in 4/12;
- source value passes in 1/12 and fails in 3/12; and
- prospective value passes in 1/12.

These are descriptive stage counts, not a common-metric ranking.

## Added failure localization

The MuJoCo invocation completes and its synthetic preflight passes, but the
registered public-source action crosses the hard deformation-orientation floor
at the native stage. It therefore never reaches complete-horizon or source-value
evaluation.

JAX-FEM and SOFA each pass their separately frozen source-physics
qualification. Their registered prediction-generation campaigns then cross a
hard orientation floor before completing all predeclared source interactions.
Both stop at full-horizon qualification, before prefix or future outcome
scoring. Partial trajectory files do not count as a completed prediction bank.

This adds a useful distinction to v1: a native continuum backend can be
implemented, pass a local source-physics gate, and still lack a valid complete
horizon. Software support, local physical plausibility, and query-level value
remain separate claims.

## Lineage and reproduction

The v2 artifact records v1 artifact
`a04edd702cc95ed1cd89fe05f3a209b036c6d1e22406161b130d89c6c56cded4`
as its parent. The added evidence files are hash-bound at their originating Git
revisions. Build with:

```bash
PYTHONPATH=src python3 -m scripts.build_cross_backend_validation_atlas_v2 \
  --output /tmp/cross-backend-validation-atlas-v2.json
```

The committed artifact is
`results/source/cross_backend_validation_atlas_v2/atlas.json`:

- artifact ID: `d1faadce843d1077e47594c17ce452acf9bdac36ce76218b4973d791a4ed7240`;
- file SHA-256: `ee6f1b90c1517dbb81ca5e73fcbfd73c0db7fe55dc271dc71207ea4858e4e421`.

Atlas v1 remains immutable. V2 is an additive public-source evidence layer and
does not authorize source-value scoring or target access for any newly added
backend.
