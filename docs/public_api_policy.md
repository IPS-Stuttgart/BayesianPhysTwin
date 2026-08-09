# Root public API policy

BayesianPhysTwin 0.4 retains a large historical convenience surface at the
package root. Removing or silently changing those imports would create avoidable
breakage for notebooks and companion repositories, while continuing to add every
new contract to the root would make later compatibility work harder.

The exact observed root surface is therefore recorded in
`api/root-public-api-v0.4.json`. The manifest is a compatibility ratchet, not a
claim that every listed research helper is a separately supported long-term API.
Normative support still follows `SUPPORT.md`, versioned provider modules,
artifact schemas, and installed command coverage.

## Policy

For the `0.4.x` line:

- existing root exports remain available in their recorded order;
- additions or removals require an explicit manifest update and compatibility
  explanation in the same pull request;
- new functionality should normally use an explicit module or namespace rather
  than expanding the package root;
- versioned provider and artifact modules remain the preferred integration
  boundary for Prob4D and Causal4D; and
- a fail-closed correction for causal leakage, provenance ambiguity, or an
  invalid scientific claim may still override the ordinary deprecation period,
  as described in `SUPPORT.md`.

A future `0.5` compatibility line may contract the root after documenting
replacement imports and deprecating supported interfaces where practical. The
`0.4` snapshot remains immutable evidence of the earlier import surface.

## Validation

From a source checkout with the package installed:

```bash
python tools/quality/check_public_api.py
python tools/quality/check_public_api.py --json
```

The checker validates the manifest schema, symbol uniqueness, literal module
identity, exact `__all__` order, existence of every exported attribute, and the
project minor-version line. The complete Python test matrix also exercises the
same contract through `tests/test_public_api_manifest.py`.
