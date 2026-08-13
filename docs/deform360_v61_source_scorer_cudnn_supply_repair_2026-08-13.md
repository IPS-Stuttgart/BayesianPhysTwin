# Deform360 v6.1 scorer cuDNN supply repair

The sole dispatch of the original v6.1 public-source scorer, workflow run
`31660983482` at protected-main revision
`9a18d3a4dd4aa95c69308f184c77958ddc4eec8d`, stopped during isolated runtime
construction. Both target-closed contract jobs passed. The self-hosted job could
not resolve the pinned `nvidia-cudnn-cu12==9.1.0.70` distribution from the
PyTorch CUDA index. Admission, scoring, retention, and artifact upload were all
skipped. No durable claim or output root was created, and no source suffix,
confirmation payload, target outcome, or held-v8 artifact was opened.

The exact Linux cuDNN `9.1.0.70` wheel remains published on PyPI. This repair
downloads that immutable wheel directly, verifies its 664,752,741-byte size and
SHA-256 digest, and installs it with `--no-deps` before applying the unchanged
CUDA runtime lock. Torch remains `2.4.0+cu121`, torchvision remains
`0.19.0+cu121`, cuDNN remains `9.1.0.70`, and gsplat remains the same
hash-locked `1.4.0+pt24cu121` wheel. The completed runtime additionally checks
the installed cuDNN distribution version.

This is a supply-route repair only. It changes no data, source cohort,
candidate artifact, model, mean, covariance, selector, endpoint algorithm,
loss, source gate, fallback, horizon, Prob4D declaration, suffix policy, or
target policy. The failed workflow is disabled and may not be retried. A
reviewed merge creates a separately named workflow and durable namespace that
may be dispatched exactly once. The unchanged source gate still decides
whether B0 is retained or a separately frozen continuation may be designed;
independent confirmation remains closed either way.
