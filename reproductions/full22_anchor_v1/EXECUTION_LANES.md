# Full-22 execution lanes

The frozen full-22 Bayesian-anchor capsule has two independent workflow lanes.

- `.github/workflows/full22-anchor-reproduction.yml` runs on a GitHub-hosted Python 3.12 environment with two workers.
- `.github/workflows/full22-anchor-reproduction-workstation.yml` runs on `workstation2` with eight workers and records the runner, CPU, and accelerator identities.

Both lanes use the exact frozen source revision
`e393bb6ff61d44815afd8d09dfc5334cb55d5524`, require the portable public-data
identity `f67534421ee2f81ec823171427fb0ac66d3ac1762eb1f5b7624ddda92d057ffc`,
validate `RunManifestV2`, and require all eight registered metric checks to pass
within the frozen absolute tolerance of `5e-7 m`.

The two executions are compared through their frozen source, portable input
identity, and numerical verification contract. Byte-identical floating-point
serialization across runtimes is not required.
