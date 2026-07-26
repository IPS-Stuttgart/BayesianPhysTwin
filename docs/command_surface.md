# Command surface and compatibility policy

Bayesian-PhysTwin has one supported command root: `bpt`.

The many installed `bpt-*` executables predate the grouped interface. They are
retained so frozen environments and tagged experiments remain reproducible, but
they are not the extension mechanism for new work.

## Registry fields

Every command is represented by `CommandSpec` in
`bayesian_phystwin.cli.command_registry` with:

| Field | Meaning |
|---|---|
| `command_id` | stable selector used by registry subcommands |
| `route` | complete grouped `bpt` route |
| `module` / `function` | lazily imported Python target |
| `description` | concise user-facing purpose |
| `legacy_alias` | retained `bpt-*` executable, or `None` for new commands |
| `status` | `stable`, `experiment`, `diagnostic`, or `archived` |
| `optional_dependencies` | package extras needed by the command |
| `owner` | protocol, contract, or milestone responsible for the command |

The machine-readable registry is available with:

```bash
bpt commands list --json
bpt commands describe COMMAND --json
```

Selectors may be a command id, grouped route, or legacy alias.

## Lifecycle statuses

### `stable`

Reusable interfaces covered by the core compatibility and installed-artifact
checks. Stable commands have direct grouped routes such as:

```bash
bpt provider manifest
bpt observation validate
bpt run manifest
bpt residual replay
bpt benchmark synthetic
```

### `experiment`

Current research protocols that may evolve with their owning versioned
milestone. They are invoked through the registry runner:

```bash
bpt experiment list
bpt experiment describe COMMAND
bpt experiment run COMMAND [arguments]
```

### `diagnostic`

Audits, comparisons, and analyses that do not by themselves define promotable
methods or paper claims:

```bash
bpt diagnostic list
bpt diagnostic run COMMAND [arguments]
```

### `archived`

Frozen historical or negative-result paths retained for exact reproduction.
They are omitted from current experiment listings:

```bash
bpt archive list
bpt archive run COMMAND [arguments]
```

Archiving a command changes its registry classification; it does not remove its
legacy executable or reinterpret a tagged experiment.

## Adding a command

1. Implement a CLI module with `main(argv: Sequence[str] | None = None)` when
   practical. The dispatcher also supports historical no-argument `main()`
   functions by temporarily presenting only the forwarded arguments in
   `sys.argv`.
2. Add one `CommandSpec` definition to the registry with a unique command id,
   status, optional extras, and owner.
3. Leave `legacy_alias=None` for new commands and do not add another
   `[project.scripts]` entry.
4. Add focused dispatch and metadata tests.
5. Document the protocol in `docs/` or the owning paper repository; keep the
   root README limited to stable and current entry points.

The packaging test compares the existing `[project.scripts]` table with the
registry mapping. This protects the frozen aliases from accidental deletion or
target drift while making additions to the top-level executable surface
conspicuous.

## Compatibility rules

- Existing `bpt-*` aliases keep their current target for frozen reproduction.
- Grouped dispatch is lazy and must not import optional experiment dependencies
  merely to render help or list commands.
- Command status is descriptive, not a claim of scientific validity. Evidence
  promotion remains governed by the owning protocol and evidence manifest.
- Operational Causal4D commands belong in the Causal4D repository. This registry
  contains only Bayesian-PhysTwin-owned interfaces and experiments.
- Paper-facing claim status belongs in `BayesianPhysTwin-Paper`, not in this
  command registry or the root README.
