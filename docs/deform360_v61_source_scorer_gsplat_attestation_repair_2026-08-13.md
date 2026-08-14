# Deform360 v6.1 gsplat attestation repair

The sole cuDNN-supply successor, GitHub Actions run `31663298362` at protected-main
revision `a671b1bafdcfb6e32ba370a2a8d4a157144651c2`, stopped during isolated-runtime
attestation. Both contract jobs passed, but the runtime guard reported
`gsplat distribution identity changed`. Admission, source scoring, bounded receipt
retention, and artifact upload were skipped. The registered durable run root, claim,
and endpoint root are absent, and the run published zero artifacts. No source suffix,
confirmation payload, target outcome, or held-v8 artifact was opened.

The checksum-pinned wheel was installed successfully and was not replaced. Direct
inspection of that exact wheel (SHA-256
`2efb8b8f4ad3275db05707fa6f9cf110482e7fd269c78a4cc7dc5b08cfc957ff`)
shows that both its distribution metadata and `gsplat/version.py` declare
`1.4.0+pt24cu121`. The guard incorrectly required the module value
`gsplat.__version__` to equal the untagged base release `1.4.0` while correctly
requiring the distribution value to equal `1.4.0+pt24cu121`.

The successor changes only that contradictory module-version assertion. It retains
the exact wheel URL, byte count, wheel hash, compiled-extension hash, install order,
Torch/CUDA/cuDNN runtime, scorer, endpoint processor, candidate artifacts, cohort,
metrics, source gate, and exact fallback. Its immutable repair contract is
`protocols/amendments/deform360_official_hub_fresh_object_session_v6_1_gsplat_attestation_runtime.json`.
The failed workflow is disabled and may not be retried; any execution uses a fresh
workflow identity and durable namespace after reviewed protected-main merge.

A target-free pre-merge reproduction installed the complete workflow dependency
stack in a disposable `/dev/shm` environment on `workstation2` without running the
scorer or creating an admission root. The reproduction passed with Python 3.10,
Torch `2.4.0+cu121`, torchvision `0.19.0+cu121`, CUDA 12.1, cuDNN distribution
`9.1.0.70`, gsplat module and distribution `1.4.0+pt24cu121`, two compute-capability
8.9 GPUs, and compiled extension SHA-256
`e0b664c9d6f355e611bdfa720103b86b399ded3dcc5ecfaf59eaade992f1359b`.
The disposable smoke script SHA-256 was
`ba87ca45ce3903d7101abc469814b4935a7767970a92474709dc31f55b83e3fc`.

This is runtime implementation evidence only. It authorizes no source result,
independent confirmation, target access, claim, deployment, or state-of-the-art
statement. The experiment continues to use public real-world Deform360 recordings,
collect no new measurements, require no human approval, exclude decoded-uniform
Prob4D fusion, and leave confirmation closed unless the registered source gate later
authorizes a separately frozen continuation.
