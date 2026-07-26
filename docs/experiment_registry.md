# Experiment registry

Bayesian-PhysTwin retains its historical `bpt-*` console scripts because frozen
commands and result manifests must continue to resolve exactly. The grouped
`bpt experiment` interface makes those research commands discoverable without
turning every historical entry point into a separately documented stable API.

## List experiments

```bash
bpt experiment list
bpt experiment list --category phystwin
bpt experiment list --json
```

Listing reads the installed `bayesian-phystwin` distribution metadata. It does
not import experiment modules, Torch, Warp, OpenCV, SciPy, or dataset adapters.
The output identifies the experiment ID, broad category, compatibility console
script, and Python target.

Current categories are:

- `phystwin`;
- `deform360`;
- `matphys`;
- `diagnostic`;
- `estimation`.

The categories are navigation metadata, not scientific evidence status. A
command being listed does not mean that its method passed a prospective gate or
supports a paper claim.

## Describe an experiment

```bash
bpt experiment describe evaluate-phystwin-official
bpt experiment describe bpt-evaluate-phystwin-official --json
```

Both the registry ID and exact compatibility script name are accepted. Describe
is also metadata-only and does not import the implementation.

## Run an experiment

```bash
bpt experiment run evaluate-phystwin-official -- --help
bpt experiment run build-phystwin-cues -- INPUT OUTPUT --case-id example
```

`run` resolves the installed console-script entry point, imports only the
selected implementation, and forwards the remaining arguments unchanged. The
optional `--` separator prevents grouped-command arguments from being confused
with experiment-specific options.

The following are intentionally unchanged:

- all installed historical `bpt-*` commands;
- their argument parsers and exit behavior;
- frozen run commands and manifests;
- experiment modules and optional dependency boundaries.

## Stable versus experimental commands

Operations with a deliberate stable grouped interface remain outside the
experiment registry:

```text
bpt provider manifest
bpt observation validate
bpt residual replay
bpt benchmark synthetic
bpt evidence summarize
bpt run manifest
```

Their legacy aliases remain installed for compatibility. The experiment
registry covers the remaining research-oriented commands and is generated from
installed package metadata, so newly added compatibility entry points become
discoverable without another handwritten command index.
