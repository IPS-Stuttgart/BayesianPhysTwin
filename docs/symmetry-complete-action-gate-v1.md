# Symmetry-complete act-or-fallback gate

## Purpose

A symmetry-complete physical twin separates an identifiable quotient from an
unresolved compact-group gauge. Prob4D can certify an action template over every
compatible group state, and Causal4D can prove that the state orbit and commanded
action use one shared transformation while bounding actuator realization error.
The final operational question is whether those pieces refer to the same
physical episode and imply sufficiently small total regret.

This module is the BayesianPhysTwin execution boundary. It independently
verifies the portable Causal4D receipt, binds all evidence identities, adds the
registered regret contributions, and returns either:

- a verified exact-optimal action;
- a verified uniformly bounded-regret action;
- a verified rejection with exact fallback; or
- an invalid-evidence failure with exact fallback.

It never selects a latent gauge representative.

## Evidence decomposition

For actions `a,b`, let Prob4D supply a complete-orbit structural upper bound

\[
\overline\Delta_{\rm struct}(a,b).
\]

This term can already include finite-hypothesis ambiguity and continuous-orbit
cover error. Let the Causal4D intervention receipt supply

\[
M_{\rm real}(a,b)
=K_a\varepsilon_a+K_b\varepsilon_b,
\]

where `epsilon_a` is the declared realization radius and `K_a` the registered
loss Lipschitz constant in the action argument. A separate calibration step may
optionally provide a symmetric target-transport margin

\[
M_{\rm transport}(a,b).
\]

The operational pairwise bound is

\[
\overline\Delta_{\rm total}(a,b)
=
\overline\Delta_{\rm struct}(a,b)
+M_{\rm real}(a,b)
+M_{\rm transport}(a,b),
\]

and the action-wise regret bound is

\[
\overline R_{\rm total}(a)
=
\max_b\overline\Delta_{\rm total}(a,b).
\]

The gate selects a deterministic minimizer of this vector. It executes the
minimizer only when

\[
\overline R_{\rm total}(a^\star)\leq\epsilon,
\]

where `epsilon` is the registered regret tolerance. Otherwise, it returns the
caller-owned fallback action exactly.

The three contributions remain separately visible in the audit record. An
actuator or transport uncertainty term cannot be hidden inside the structural
certificate.

## Independent Causal4D receipt verification

The consumer does not import Causal4D. It expects the portable version-1 receipt
and independently checks:

- exact field set and schema version;
- the content-derived SHA-256 receipt identity;
- contract and provenance identifiers;
- unique ordered group-element identifiers;
- action-bank and orbit dimensions;
- observed realization radii no larger than declared radii;
- nonnegative action-loss Lipschitz constants;
- the identity `m_a = K_a epsilon_a`;
- the pairwise identity `m_ab = m_a + m_b` with zero diagonal;
- registered radius scope.

The command and realization arrays are retained in the receipt hash, even though
the gate needs only their dimensions and exported margin. Consequently, changing
the physical command, measured realization, declared radius, or provenance
invalidates the receipt.

## Cross-context binding

A cryptographically valid receipt is not sufficient if it belongs to another
physical context. Before acting, the gate requires exact equality of:

- state-evidence identity;
- action-template-bank identity;
- loss-program identity;
- fallback identity.

The structural certificate has its own identifier, and a target-transport margin
is accepted only with an explicit calibration-certificate identifier. Mismatched
or missing identities produce `invalid-fail-closed`, not an optimistic repair.

## Four status outcomes

### `verified-exact-optimal`

The minimax action has nonpositive pairwise upper bounds against every
competitor. It is optimal for every model covered by the registered structural,
realization, and transport assumptions.

### `verified-bounded-regret`

The action is not exactly optimal for every covered model, but its worst-case
regret does not exceed the registered tolerance.

### `verified-reject-exact-fallback`

All evidence is internally valid, but the smallest total regret exceeds the
tolerance. The exact caller-owned fallback is returned.

### `invalid-fail-closed`

The receipt is malformed, tampered, cross-context, insufficient in scope, or
otherwise invalid. No minimax claim is produced; the exact fallback is returned.
This status is distinct from a scientifically valid rejection.

## Controlled end-to-end study

The registered structural matrix corresponds to ideal action losses `(0,2,1)`:

\[
\overline\Delta_{\rm struct}
=
\begin{bmatrix}
0 & -2 & -1\\
2 & 0 & 1\\
1 & -1 & 0
\end{bmatrix}.
\]

The Causal4D receipt gives action 0 realization radii
`0.0, 0.2, 0.8, 1.0, 1.2`, with unit loss Lipschitz constant. At zero regret
tolerance:

- radii through `1.0` retain exact optimality of action 0;
- radius `1.2` raises its regret upper bound to `0.2` and triggers exact fallback.

At a separately registered tolerance `0.25`, the radius-`1.2` case executes
action 0 as explicitly bounded regret.

A separate transport margin raises the action-0-versus-fallback bound by `1.1`.
Although both the structural and intervention receipts remain valid, this target
term forces fallback. Two adversarial controls also fail closed:

- changing a declared radius while retaining the old receipt identifier;
- presenting a content-valid receipt bound to another state-evidence identity.

The study is deterministic and its result must be byte-identical across two
executions.

## Relationship to Prob4D and Causal4D

The complete chain is:

1. **Prob4D:** preserve `p(quotient, group)` without invented gauge information
   and compute complete-orbit pairwise structural bounds.
2. **Causal4D:** bind state and action to one registered group transform, verify
   the command orbit, and emit a content-addressed realized-intervention receipt.
3. **BayesianPhysTwin:** independently verify the receipt, bind all identities,
   add structural, realization, and transport terms, and execute or restore exact
   fallback.

This decomposition assigns a falsifiable responsibility to each repository. A
single model cannot silently assert that its own state symmetry, actuator
transform, target calibration, and decision are all valid.

## Claim boundary

The gate proves a conditional composition statement. It does not establish that
the physical group is correct, that a learned provider exposes the full orbit,
that the action representation is valid for an unseen robot/object, that a
realization radius or transport margin has target coverage, or that an admitted
action is safe. Those assumptions require separate empirical or statistical
certificates.

The decisive real-data promotion is an object- or recording-disjoint experiment
that calibrates the realization and transport margins before opening target
outcomes. Until then, this is a verified systems mechanism rather than a
deployment guarantee.
