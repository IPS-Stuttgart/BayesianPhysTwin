# Anytime-valid simulator admission — implementation and evidence status

## Current state

The branch now contains a complete four-level admission hierarchy and a
fail-closed version-4 deployment controller.

1. **Version 1:** delayed paired outcomes, content-addressed contracts,
   geometric alpha spending across declared epochs, degradation monitoring,
   and exact caller-owned physical fallback.
2. **Version 2:** a shared-alpha intersection--union rule for jointly requiring
   positive mean utility and a harmful-update rate below a registered ceiling.
   It is more efficient than splitting alpha, but one fixed component null must
   remain valid throughout an admission epoch.
3. **Version 3:** a minimum-score certificate valid under a pointwise switching
   union null. The reason for rejecting the candidate may change at every
   reveal.
4. **Version 4:** a lower envelope of independently tuned component e-factors.
   It keeps the switching-null guarantee while removing the version-3
   restriction that utility and harm share one scalar betting parameter.

The deployable controller is
`src/bayesian_phystwin/anytime_factor_envelope_controller_v4.py`. It binds the
candidate, fallback, score, harm definition, information set, reveal policy,
factor family, and factor grids into one SHA-256 decision-contract identity.
It supports predictable issue and delayed resolution of paired trials,
transactional reveal validation, closed-epoch outcome retention without reuse,
minimum-sample admission, epoch-wise alpha spending, and exact-object fallback
selection.

## Main theorem now implemented

For component nulls with registered nonnegative conditional e-factors
\(F_{t,k,\theta_k}\), assume only that at every reveal at least one component
null is valid. The active component may change arbitrarily with the observed
past. For every fixed parameter tuple, define

\[
L_{t,\theta}=\min_k F_{t,k,\theta_k}.
\]

The lower envelope is dominated by the factor belonging to whichever component
null is active, so

\[
\mathbb E[L_{t,\theta}\mid\mathcal F_{t-1}]\le 1.
\]

Products over time and any outcome-independent mixture over fixed parameter
tuples therefore remain e-processes. The implementation specializes this result
to a bounded utility factor and an exact Bernoulli harmful-rate likelihood
ratio, mixed over a frozen Cartesian parameter grid.

The proof and claim boundary are documented in
`docs/anytime_factor_envelope_v4.md`.

## Sealed controlled evidence

All reported numbers below come from committed `result.json` artifacts rather
than workflow log summaries.

| Study | Registered result | Most informative findings |
|---|---|---|
| Shared-alpha IUT v2 | `results/science/anytime_joint_admission_v2/result.json` | Maximum null Wilson upper bound **0.01657**. Moderate-case power **0.8999** with shared alpha versus **0.8501** after Bonferroni splitting, a **+0.0498** gain. Median first crossing fell from **242** to **214** observations. Strong-case power was **1.0000** for both methods. |
| Switching certificate v3 | `results/science/anytime_switching_admission_v3/result.json` | Maximum robust-null Wilson upper bound **0.01281**. In the adversarial changing-failure-mode stream, the out-of-scope latched v2 rule crossed in **1.0000** of replications, while the switching-valid minimum-score rule crossed in **0.0000**. Moderate robust power was **0.7359** and strong power **1.0000**. |
| Factor envelope v4 | `results/science/anytime_factor_envelope_v4/result.json` | Maximum null Wilson upper bound **0.01017** and switching-null crossing **0.0000**. On the frozen confirmation roster, moderate power rose from **0.7430** for v3 to **0.8334** for v4, a **+0.0904** gain. The median crossing ratio was **1.1111**, within the preregistered 1.15 ceiling, and strong power was **1.0000**. |

Every registered mechanism gate passed.

## Evidence provenance and custody

The version-4 scenario families were inherited from the already observed
version-3 study; this is disclosed rather than presented as a fresh scenario
selection. A separate version-4 pilot roster with seed base `2026090400` was
used only to choose the fixed gates. The retained confirmation roster used seed
base `2026091400`, which was frozen before that roster was opened. No real
outcomes were used.

The evidence workflow verified:

- the exact protocol SHA-256 identity;
- the frozen confirmation seed base and conjunctive gates;
- an expected lower-envelope factor no larger than one in every registered null
  phase;
- canonical LF-only CSV serialization;
- all artifact checksums in `SHA256SUMS`; and
- removal of the one-shot execution workflow after publication.

## Software validation

The latest integrated-controller validation passed **55 focused tests** covering
versions 1--4, exact fallback identity, delayed-outcome order, transactional
reveal handling, changing failure modes, contract hashing, and stable-suite
registration. Ruff formatting and static checks, Python compilation, and the
central test-suite manifest check also passed.

During this work, two nontrivial defects were found and fixed:

1. a version-3 test incorrectly required an e-process never to rise above one
   on a fixed null-compatible path; the corrected test checks the registered
   stopping boundary, which is the actual guarantee; and
2. version-3 reveal handling could consume a pending trial before validating a
   malformed loss. Reveal validation is now atomic and retryable.

All version-2 through version-4 tests are registered in
`.github/quality/test-suites.json` under `stable-core-coverage`.

## Claim authorized by the current evidence

The current branch supports the following controlled-method claim:

> A conjunction of simulator-admission requirements can be monitored with one
> anytime-valid e-process under a pointwise union of conditional nulls. Taking
> the lower envelope of independently tuned component e-factors remains valid
> even when the active failure mode changes over time. In the registered
> controlled confirmation, independent tuning recovered 9.04 percentage points
> of moderate-case power relative to the shared-parameter switching certificate
> while retaining the registered null controls.

The current evidence does **not** authorize claims of physical safety, causal
identification, universal distribution-shift robustness, posterior calibration,
real-object superiority, or state-of-the-art task performance.

## Remaining decisive empirical gate

The frozen fresh recursive-corruption roster and any real-data shadow stream
remain unexecuted for claim-bearing admission. A prospective real validation
must keep the candidate, exact fallback, score, harmful-outcome definition,
information set, reveal order, factor grids, alpha schedule, and epoch policy
fixed before opening the corresponding outcomes.

Until such a stream is completed, the strongest result is a theorem plus
sealed controlled mechanism evidence and a validated deployable controller—not
a fresh real-world admission result.
