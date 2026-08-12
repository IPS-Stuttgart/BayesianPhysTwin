# Public API policy

BayesianPhysTwin 0.4 separates two compatibility surfaces with different
purposes:

1. the large historical package-root convenience API; and
2. the deliberately small `bayesian_phystwin.v1` integration API used by new
   ecosystem consumers.

Both surfaces have exact, ordered export snapshots. The snapshots are
compatibility ratchets, not claims that every research helper has the same
support level.

## Historical package-root surface

BayesianPhysTwin 0.4 retains a large historical convenience surface at the
package root. Removing or silently changing those imports would create avoidable
breakage for notebooks and companion repositories, while continuing to add every
new contract to the root would make later compatibility work harder.

The exact observed root surface is recorded in
`api/root-public-api-v0.4.json`.

For the `0.4.x` line:

- existing root exports remain available in their recorded order;
- additions or removals require an explicit manifest update and compatibility
  explanation in the same pull request;
- new functionality should normally use an explicit module or namespace rather
  than expanding the package root; and
- a fail-closed correction for causal leakage, provenance ambiguity, or an
  invalid scientific claim may still override the ordinary deprecation period,
  as described in `SUPPORT.md`.

The legacy names are implemented as a lazy compatibility shim. Importing
`bayesian_phystwin` alone does not import the research, Deform360, graph, vision,
or experiment modules that own those names. Accessing a recorded root export
imports its owning module once, returns the original object, and caches that
object on the package. The symbol set, order, and object identities therefore
remain compatible while the default import path becomes NumPy-only and cheap.

The ownership table is recorded in `api/root-export-migration-v1.json`. It maps
every frozen `0.4` root symbol to the explicit module import that new code should
use and binds the ordering to `api/root-public-api-v0.4.json`. The table targets
the planned `0.5` compatibility line, but it does not itself remove a root name,
activate a deprecation warning, or authorize a version change.

A future `0.5` compatibility line may contract the root after documenting
replacement imports and deprecating supported interfaces where practical. The
`0.4` snapshot remains immutable evidence of the earlier import surface.

## Versioned integration surface

New Prob4D, Causal4D, and independent integrations should prefer
`bayesian_phystwin.v1`. Its exact ordered surface is recorded in
`api/versioned-public-api-v1.json` and is intentionally much smaller than the
package root.

Within the BayesianPhysTwin `0.4.x` line:

- the recorded `bayesian_phystwin.v1.__all__` order and symbol set are frozen;
- a compatible implementation may fix behavior behind those contracts without
  changing their documented semantics;
- adding an export requires an explicit snapshot update and justification that
  it belongs to the long-lived integration boundary;
- removing or semantically changing an export requires a new versioned
  namespace rather than silently altering `v1`; and
- research estimators, experiment runners, and optional candidate methods stay
  in explicit modules unless a demonstrated consumer need justifies promotion.

Provider-specific modules and artifact schemas may remain narrower than
`bayesian_phystwin.v1`. A green API check establishes compatibility only; it is
not evidence of estimator accuracy, covariance calibration, physical transfer,
or deployment safety.

## Validation

From a source checkout with the package importable:

```bash
python tools/quality/check_public_api.py
python tools/quality/check_public_api.py \
  --manifest api/versioned-public-api-v1.json
python tools/quality/check_root_export_migration.py
```

Add `--json` to any checker for a machine-readable report. The public-API
checker validates the manifest schema, schema-policy pairing, symbol uniqueness,
literal module identity, exact `__all__` order, existence of every exported
attribute, and the project minor-version line. The migration checker additionally
validates exact root-snapshot coverage, runtime owner mappings, and object
identity between every lazy root export and its explicit owning module.

The complete Python test matrix exercises both snapshots, isolated root and
`v1` import boundaries, and the migration map. `MANIFEST.in` also requires all
three manifest files and both checkers in the source distribution.
