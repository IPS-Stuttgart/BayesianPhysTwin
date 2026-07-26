# Command-line interface

Bayesian-PhysTwin installs exactly one executable:

```bash
bpt
```

The single entry point separates the stable operational surface from the larger
research-command inventory. This keeps package installation predictable and
prevents every experiment module from becoming an accidental compatibility
promise.

## Stable commands

The stable routes are:

```text
bpt provider manifest
bpt observation validate
bpt residual replay
bpt benchmark synthetic
bpt evidence summarize
bpt run manifest
```

Use `--help` after any complete route to inspect its arguments.

## Registered research commands

Non-stable research commands are available through a lazy registry:

```bash
bpt experiment list
bpt experiment describe phystwin-refit
bpt experiment run phystwin-refit --help
```

`list` and `describe` do not import the experiment module. `run` imports only the
selected module and forwards every remaining argument. The dispatcher supports
both modern `main(argv=None)` functions and historical `main()` functions that
read `sys.argv`.

Registry membership means that a command remains executable; it does not promote
the command or its outputs to a stable API or a scientific result.

## Migration from removed executables

All `bpt-*` console-script entry points have been removed. The five former stable
executables map directly to grouped routes:

```text
bpt-provider-manifest              -> bpt provider manifest
bpt-validate-observation-belief    -> bpt observation validate
bpt-replay-residuals               -> bpt residual replay
bpt-synthetic-benchmark            -> bpt benchmark synthetic
bpt-run-manifest                   -> bpt run manifest
```

Every other removed executable maps to its registry identifier by dropping the
`bpt-` prefix:

```text
bpt-phystwin-refit --epochs 20
-> bpt experiment run phystwin-refit --epochs 20

bpt-confirm-phystwin-bayesian-anchor ...
-> bpt experiment run confirm-phystwin-bayesian-anchor ...
```

Removed executable names are not accepted as command aliases. Use the exact
registry identifier shown by `bpt experiment list`.

Historical manifests and frozen result artifacts may retain old command strings
as provenance. They are records of the original execution and should not be
rewritten merely because the current invocation surface changed.
