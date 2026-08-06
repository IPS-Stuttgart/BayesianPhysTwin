# Command-line interface

Bayesian-PhysTwin installs exactly one executable:

```bash
bpt
```

The grouped interface separates stable operations from research workflows and
prevents every experiment module from becoming an accidental compatibility
promise.

## Stable routes

```text
bpt provider manifest
bpt ecosystem validate
bpt observation validate
bpt residual replay
bpt benchmark synthetic
bpt evidence summarize
bpt run manifest
```

Use `--help` after any complete route to inspect its arguments. Stable routes
are lazily imported, so root help and registry inspection remain NumPy-only.

`bpt ecosystem validate` checks installed BayesianPhysTwin, Prob4D, and Causal4D
package lines against the bundled three-repository compatibility lock. Exact
source commits may be supplied explicitly; the command performs no network
resolution. See [ecosystem compatibility](ecosystem_compatibility.md).

## Canonical registry

Every command is represented by a `CommandSpec` in
`bayesian_phystwin.cli.command_registry`.

| Field | Meaning |
| --- | --- |
| `command_id` | Exact selector used by lifecycle runners. |
| `route` | Complete grouped `bpt` route. |
| `previous_routes` | Former grouped routes retained only for inspection and migration. |
| `module` / `function` | Lazily imported Python target. |
| `description` | Concise user-facing purpose. |
| `legacy_alias` | Removed historical `bpt-*` name, or `None`. |
| `status` | `stable`, `experiment`, `diagnostic`, or `archived`. |
| `optional_dependencies` | Package extras required by the command. |
| `owner` | Contract, protocol, or milestone responsible for the command. |

Inspect the complete registry in human-readable or JSON form:

```bash
bpt commands list
bpt commands list --status diagnostic --json
bpt commands describe confirm-phystwin-bayesian-anchor
bpt commands describe bpt-phystwin-refit --json
```

`commands describe` accepts a command ID, current or former grouped route, or
removed legacy alias for inspection. This does not make a historical selector
executable.

## Lifecycle catalogs

### Current experiments

```bash
bpt experiment list
bpt experiment describe confirm-phystwin-bayesian-anchor
bpt experiment run confirm-phystwin-bayesian-anchor --help
```

These are active research protocols. Their behavior may evolve with the owning
versioned protocol.

### Diagnostics

```bash
bpt diagnostic list
bpt diagnostic describe audit-phystwin-calibration
bpt diagnostic run audit-phystwin-calibration --help
```

Diagnostics are audits, comparisons, and analyses that do not by themselves
define promotable methods or paper claims.

### Archived paths

```bash
bpt archive list
bpt archive describe evaluate-phystwin-state-injection
bpt archive run evaluate-phystwin-state-injection --help
```

Archived entries preserve historical or negative-result workflows. Their
presence is an executable reproduction aid, not a statement of current
scientific preference.

## Migration from removed executables

The current package does not install any `bpt-*` executable. Map a historical
name or former grouped route to its current route with:

```bash
bpt commands migrate bpt-provider-manifest
# bpt provider manifest

bpt commands migrate bpt-phystwin-refit
# bpt experiment run phystwin-refit

bpt commands migrate bpt experiment run audit-phystwin-calibration
# bpt diagnostic run audit-phystwin-calibration
```

Removed names and former grouped routes are rejected by `experiment run`,
`diagnostic run`, and `archive run`; only exact current registry IDs are
executable.

Frozen releases and tags retain their original entry points. Existing result
artifacts and manifests may also retain old command strings as immutable
provenance. Do not rewrite those records merely because the current invocation
surface changed.

## Lazy dispatch and optional dependencies

Listing or describing commands never imports their implementation modules.
`run` imports only the selected target. Historical `main()` functions are
adapted by setting `sys.argv[0]` to the canonical grouped route; modern
`main(argv)` and `main(*, argv=...)` functions receive only forwarded
arguments.

When a declared optional dependency is absent, the dispatcher reports the
relevant package extra. Internal `ModuleNotFoundError` failures are re-raised
rather than being mislabeled as installation problems.

## Adding a command

1. Implement a CLI module with `main(argv=None)` when practical.
2. Add one `CommandSpec` with a unique ID and route, lifecycle status, optional
   extras, owner, and concise description.
3. Set `legacy_alias=None` for new commands.
4. Do not add another `[project.scripts]` entry.
5. Add focused dispatch, registry, and installed-artifact tests.
6. Document the protocol in `docs/` or the owning paper repository.

Registry status is descriptive, not evidence promotion. Method freezing, split
integrity, target sealing, statistical analysis, and claim review remain
separate requirements.
