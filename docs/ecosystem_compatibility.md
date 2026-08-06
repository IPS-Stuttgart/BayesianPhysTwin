# Ecosystem compatibility lock

BayesianPhysTwin, Prob4D, and Causal4D evolve independently, while frozen
experiments continue to require exact source revisions. The bundled
`ecosystem_compatibility_v1.json` records one installed-wheel combination that
passed the public three-repository integration path. It is a development
compatibility reference, not a replacement for revisions and artifact digests
bound by a scientific protocol.

## Validate an installation

The base package exposes a NumPy-free stable command:

```bash
bpt ecosystem validate
bpt ecosystem validate --json
bpt ecosystem validate --require-all --exact-versions
```

By default, BayesianPhysTwin is required and absent companion packages are
reported as optional. `--require-all` requires all three packages. Compatible
minor lines and the exact versions used by the lock are reported separately;
`--exact-versions` requires the latter.

An orchestration environment can additionally verify exact checked-out commits:

```bash
bpt ecosystem validate --require-all --json \
  --revision prob4d="$(git -C ../Prob4D rev-parse HEAD)" \
  --revision causal4d="$(git -C ../Causal4D rev-parse HEAD)"
```

Selectors `bpt`, `bayesian-phystwin`, `prob4d`, and `causal4d` are accepted.
Revisions must be literal lowercase 40-character Git commits. A supplied source
revision is compared exactly; it is never coerced, shortened, or resolved over
the network.

Use `--output-json PATH` to persist the content-addressed report. The command
returns zero only when every required installed package, version condition, and
supplied revision passes.

## CI policy

The blocking Causal4D provider workflow resolves its default consumer revision
from the committed lock. Pull requests and pushes therefore test a deterministic
known-good consumer rather than whichever commit happens to be at Causal4D
`main` when a runner starts.

A scheduled or explicitly requested latest-main canary remains present and is
allowed to fail without invalidating a compatible BayesianPhysTwin change. Its
purpose is early warning for ecosystem drift. A canary failure should lead to a
separate compatibility update in the affected repository and, after the public
installed-wheel path passes, a reviewed lock update.

Manual workflow dispatch may override the locked Causal4D ref for diagnosis. An
override is clearly reported and is not presented as verification of the
committed exact revision.

## Updating the lock

1. Run the public installed-wheel path with explicit Prob4D and Causal4D commits.
2. Require the complete producer-to-BayesianPhysTwin-to-Causal4D tests to pass.
3. Record exact package versions, repository commits, Python version, workflow
   run, and test count.
4. Update the bundled JSON in an ordinary reviewable pull request.
5. Run the lock parser, CLI, distribution, and both locked/canary workflow tests.

A green compatibility lock establishes only interface and packaging
interoperability. It is not evidence of observation quality, calibrated
uncertainty, physical prediction benefit, intervention benefit, or state of the
art.
