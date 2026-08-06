# Ecosystem compatibility lock

BayesianPhysTwin, Prob4D, and Causal4D evolve independently, while frozen
experiments continue to require exact source revisions. The bundled
`ecosystem_compatibility_v1.json` records one installed-wheel combination that
passed the public three-repository integration path. It is a development
compatibility reference, not a replacement for revisions and artifact digests
bound by a scientific protocol.

The version-1 lock binds the exact source checkouts exercised by workflow run
`31019529164`:

- BayesianPhysTwin `3f37fbc87975f0581a0e58434e53b44c4d61b402`;
- Prob4D `9ad07f89f9a85b68cf1375a4087ffa447b6af846`; and
- Causal4D `b0bf0c2de176b29534ef59484ad167b8f27d9dae`.

The BayesianPhysTwin value is the exact pull-request merge revision checked out
by that successful run. Later squash or development commits are not substituted
into the historical lock merely because they contain similar changes.

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
bpt ecosystem validate --require-all --exact-versions --json \
  --revision bpt="$(git -C ../BayesianPhysTwin rev-parse HEAD)" \
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

Three complementary lanes use the committed lock:

1. `Causal4D provider compatibility` tests current BayesianPhysTwin provider
   modules and downstream behavior against the locked Causal4D revision.
2. `Locked ecosystem installed-wheel compatibility` builds the current reviewed
   BayesianPhysTwin source with the exact locked Prob4D and Causal4D companions.
   This is the blocking forward-compatibility gate for a BayesianPhysTwin change.
3. The same workflow separately reproduces the complete historical trio at all
   three locked revisions, proving that the lock still names the source that
   produced its evidence.

Pull requests and pushes therefore have deterministic known-good companion
revisions rather than whichever commits happen to be at moving `main` branches
when a runner starts. The locked workflow verifies its literal workflow pins
against the packaged JSON before any cross-repository build begins.

A scheduled or explicitly requested latest-main canary tests current Prob4D and
Causal4D `main`. It is a job-level `continue-on-error` lane and is allowed to
fail without invalidating a compatible BayesianPhysTwin change. Its purpose is
early warning for ecosystem drift. A canary failure should lead to a separate
compatibility update in the affected repository and, after the public
installed-wheel path passes, a reviewed lock update.

The older broad `Three-repository installed-wheel golden path` remains useful
for repository-dispatch and explicit branch testing. It is not the source of
truth for the exact version-1 lock unless all three recorded checkouts match the
lock.

## Updating the lock

1. Run the public installed-wheel path with explicit BayesianPhysTwin, Prob4D,
   and Causal4D commits.
2. Require the complete producer-to-BayesianPhysTwin-to-Causal4D tests to pass.
3. Record the exact checked-out revisions from the Actions logs, not a later
   squash, branch name, or inferred equivalent commit.
4. Record package versions, Python version, workflow run, and test count.
5. Update the bundled JSON and literal workflow pins in one reviewable pull
   request.
6. Run the lock parser, CLI, wheel/sdist, provider, historical reproduction,
   current-source/locked-companion, and latest-main canary policy tests.

A green compatibility lock establishes only interface and packaging
interoperability. It is not evidence of observation quality, calibrated
uncertainty, physical prediction benefit, intervention benefit, or state of the
art.
