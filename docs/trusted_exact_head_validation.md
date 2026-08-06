# Trusted exact-head validation on a self-hosted runner

The `Trusted exact-head validation` workflow replaces one-off maintenance pull
requests that hard-code several unrelated PR heads. It is dispatched from the
default branch and accepts two operator-reviewed inputs:

- the open same-repository pull-request number; and
- the exact 40-character head SHA visible during review.

Before any PR source is checked out, the default-branch control plane queries the
GitHub API and requires the PR to be open, based on `main`, hosted in this same
repository, and still at the supplied SHA. The workflow then checks out exactly
that commit with persisted credentials disabled, runs full source checks and the
complete test suite, builds and smoke-tests a wheel, verifies a clean tree, and
uploads compact environment and wheel-digest evidence. It never writes to the
branch.

## Required repository setup

Create and protect the GitHub environment named
`trusted-self-hosted-validation` before using the workflow:

1. require an explicit reviewer who is independent of the PR head change;
2. do not attach repository or environment secrets;
3. limit deployment branches to the default branch workflow;
4. retain the workflow's read-only token permissions; and
5. verify the displayed PR number and exact SHA again at environment approval.

Running PR-controlled packaging and tests on a self-hosted runner is privileged
code execution. Same-repository admission plus manual dispatch are not a
substitute for environment approval.

## Dispatch procedure

From the Actions page on `main`, select `Trusted exact-head validation`, enter
the PR number and the full current head SHA, review the environment approval,
and run it. If the head moves, the workflow fails before checkout and must be
dispatched again with the newly reviewed SHA.

The uploaded artifact is validation evidence for that exact revision only. It
is not empirical accuracy evidence and does not authorize target-data access,
method retuning, or a scientific claim.
