# Deform360 v6 Ninja build-tool repair - 2026-08-12

## Retained failure

Protected-main run `31576200607` completed the checksum-pinned CUDA 12.1.1
bootstrap and verified the registered GNU 11.5.0 host compiler. It then stopped
before source prediction because PyTorch's gsplat extension loader could not
find the Ninja build tool.

The bounded receipt reports terminal stage
`build-isolated-gpu-source-runtime`, `0/10` physical manifests, `0/100` source
prediction seals, and false values for every suffix, confirmation, target,
replacement, and claim boundary.

## Correction

The additive amendment installs exactly `ninja==1.13.0` from the Linux x86_64
wheel using `--ignore-installed`, `--no-deps`, `--only-binary=:all:`, and
`--require-hashes`. It binds:

- wheel filename and SHA-256;
- installed distribution version;
- runtime-relative executable path;
- installed executable SHA-256;
- exact executable version output; and
- `torch.utils.cpp_extension.is_ninja_available()` returning true.

The runtime's `bin` directory is explicitly prepended to `PATH` in the build
step and published through `GITHUB_PATH` for the later prediction step. Both
technical-failure and successful receipts distinguish the registered build-tool
identity from the identity actually observed at runtime.

## Boundary

This is build tooling for the already-declared gsplat backend. It changes no
candidate, cohort, camera panel, model, loss, covariance, gate, fallback,
suffix policy, target policy, or claim. A new protected-main execution must
still satisfy the unchanged prediction-first source protocol.
