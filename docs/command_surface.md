# Command surface

Bayesian-PhysTwin has one supported command namespace:

```bash
bpt <namespace> <command> [arguments]
```

The declarative registry in
`src/bayesian_phystwin/cli/command_registry.py` is the source of truth for every
grouped command. Each entry records:

- a stable command ID and grouped route;
- the implementation target;
- lifecycle status (`stable`, `experiment`, `diagnostic`, or `archived`);
- the owning protocol or milestone;
- optional package extras; and
- the frozen historical `bpt-*` compatibility alias, when one exists.

Inspect it from an installed package:

```bash
bpt commands
bpt commands --status diagnostic
bpt commands --all
bpt commands --all --format json
```

The default listing contains only stable and current experimental commands.
Diagnostics remain available for audits and mechanism localization. Archived
commands remain available solely to reproduce frozen historical runs.

## Stable routes

| Grouped route | Frozen compatibility alias | Purpose |
| --- | --- | --- |
| `bpt run manifest` | `bpt-run-manifest` | Create or validate content-addressed run provenance. |
| `bpt provider manifest` | `bpt-provider-manifest` | Emit the versioned Causal4D provider capability manifest. |
| `bpt observation validate` | `bpt-validate-observation-belief` | Validate or summarize an `ObservationBeliefV1` artifact. |
| `bpt residual replay` | `bpt-replay-residuals` | Replay exported residuals through the robust likelihood. |
| `bpt benchmark synthetic` | `bpt-synthetic-benchmark` | Run the controlled fixed-graph benchmark. |
| `bpt evidence summarize` | none | Summarize matched guarded prospective evidence. |

`bpt evidence summarize` demonstrates the policy for new commands: it is
available only through the grouped namespace and does not create another
console script.

## Experiment, diagnostic, and archived routes

Lifecycle commands use a uniform ID-based interface:

```bash
bpt experiment list
bpt experiment run <id> [arguments]

bpt diagnostic list
bpt diagnostic run <id> [arguments]

bpt archived list
bpt archived run <id> [arguments]
```

The command ID is stable registry metadata. The compatibility alias, when
present, invokes the same implementation target.

## Compatibility policy

The 79 historical `bpt-*` aliases are frozen. Their exact names and targets are
checked against `pyproject.toml`, and the sorted alias-name fingerprint is
checked at import time and in the test suite. This protects frozen scripts and
result manifests from accidental alias removal, renaming, or retargeting.

New commands must be grouped-only:

1. implement the command under `bayesian_phystwin.cli`;
2. add one `CommandSpec` with `legacy_alias=None`;
3. declare its lifecycle status, milestone, and optional extras;
4. add focused semantic tests for the command itself; and
5. do not add a new entry under `[project.scripts]`.

Registry tests compare the installed console scripts with the frozen aliases.
The local smoke suite derives stable help checks directly from the registry, so
a new stable command cannot silently disappear from package validation.

## Lifecycle meanings

`stable`
: Versioned package contracts and controlled benchmarks intended for routine
  use.

`experiment`
: Current frozen or prospective research protocols visible in the primary
  command surface.

`diagnostic`
: Audits, ablations, mechanism-localization tools, and non-promotable analyses.

`archived`
: Historical experiment entry points retained solely for reproducibility.
