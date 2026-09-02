# Exact decision-directed probing under quotient ambiguity

## Setting

Let `H={1,...,m}` be a finite registered physical hypothesis set with prior
weights `p`. Positive prior mass defines the admissible support. An observation
determines only posterior masses `lambda_c` over a registered partition
`C_1,...,C_K`; it does not determine the conditional distribution inside each
class. The compatible complete-belief set is

\[
\mathcal Q(\lambda)=\left\{q\ge 0:
q_i=0\text{ if }p_i=0,\quad
\sum_{i\in C_c}q_i=\lambda_c\right\}.
\]

For finite direct actions `a in A`, let `L(i,a)` be the registered loss under
hypothesis `i`. A finite probe `e` has outcomes `z in Z_e`, cost `c_e`, and
registered likelihood `O_e(i,z)`. Before observing its outcome, a contingent
policy `delta: Z_e -> A` is fixed.

## A common class of direct and probe meta-actions

A direct action and a probe-contingent policy must be compared against the same
alternatives. Define the hypothesis-wise meta-action loss

\[
G(i,m)=
\begin{cases}
L(i,a), &m=a\text{ is a direct action},\\
c_e+\sum_z O_e(i,z)L(i,\delta(z)),
  &m=(e,\delta)\text{ is a probe policy}.
\end{cases}
\]

Probe cost is therefore represented in the same loss units and in the same
comparison matrix as every direct action and every other probe.

This union is not optional. Worst-case direct-action regret is measured against
the best direct action for each complete belief, while within-probe regret is
measured against the best contingent policy using that probe. Adding a probe
cost to one of these two internally normalized regrets does not make them
comparable. Act-versus-probe selection requires one common meta-action class.

## Theorem: exact robust regret over the union

For any direct or probe meta-action `m`, its worst-case ex-ante regret over every
compatible complete belief is

\[
\overline{\operatorname{Reg}}(m)
=
\max_{m'}
\sum_c \lambda_c
\max_{i\in C_c:p_i>0}
\bigl[G(i,m)-G(i,m')\bigr].
\]

Consequently, minimizing this expression over the finite union gives the exact
minimax direct-or-probe meta-action. A registered tolerance determines whether
that minimax choice is admitted. When no meta-action satisfies the tolerance,
the implementation returns the separately registered fallback action without
calling it certified.

### Proof

For a fixed complete belief,

\[
R(q,m)-\min_{m'}R(q,m')
=
\max_{m'}\sum_i q_i\bigl[G(i,m)-G(i,m')\bigr].
\]

The comparator set is finite, so supremum and maximum commute. For fixed
`m,m'`, the objective is linear in `q`. The constraints on `q` separate by
quotient class. Within each class, its fixed mass can concentrate on any
prior-supported maximizing hypothesis. The support function is therefore
exactly the displayed class-wise weighted maximum. Minimization over the finite
candidate meta-actions completes the result.

## Probe-only corollary

For a fixed probe, direct actions and other probes may be removed from the union.
Writing the candidate and comparator as contingent policies `delta,delta'`
gives

\[
\max_{\delta'}
\sum_c \lambda_c
\max_{i\in C_c:p_i>0}
\sum_z O_e(i,z)
\left[L(i,\delta(z))-L(i,\delta'(z))\right].
\]

The probe cost cancels inside this fixed-probe comparison. This within-probe
quantity is useful for diagnosing the best contingent policy, but it must not be
ranked against direct-action regret. The executable act/probe certificate uses
the common union instead.

## Consequences

### Direct actions are the one-outcome special case

When a zero-cost probe has one certain outcome, every contingent policy is one
direct action. Its expected policy-loss matrix reproduces the direct loss matrix
exactly.

### A decision can become identifiable without identifying the state

Consider two supported physical hypotheses in one quotient class and two
opposing actions. No direct action has zero robust regret. A binary probe that
perfectly separates the hypotheses has contingent policy `(action_0, action_1)`.
At zero cost, that meta-action has zero robust regret even though the pre-probe
physical state remains unidentified.

### A direct-only certificate can be invalidated by adding probes

An action may be optimal for every compatible complete belief *among direct
actions* while a probe-contingent policy weakly improves it for every belief and
strictly improves it for some. Therefore a router that checks direct actions
first and probes only after direct failure is not generally optimal or exact.
The implementation certifies the complete registered union at once.

### An uninformative probe can still randomize

If probe outcomes have the same distribution under every hypothesis, they carry
no state information. A deterministic outcome-contingent policy can nevertheless
use the outcome as exogenous randomization and lower minimax regret. Empirical
claims of decision-directed information value must therefore include an
uninformative or likelihood-scrambled probe control, rather than comparing only
with deterministic no-probe actions.

## Act, probe, or fallback

The implementation enumerates every registered direct action and every
contingent policy of each finite probe, constructs their common
hypothesis-by-meta-action loss matrix, and runs the exact quotient certificate
on that union.

1. **Act:** the minimax admitted meta-action is a direct action.
2. **Probe:** the minimax admitted meta-action is one probe and one frozen
   outcome-to-action policy.
3. **Fallback:** no union meta-action satisfies the registered regret tolerance,
   so the caller's fallback action is returned without a certification claim.

Direct actions precede probe policies in the union, so exact numerical ties are
deterministically resolved in favor of the lowest-index direct action. Probe and
policy indices are also retained explicitly.

## Computational boundary

For `A` actions and `Z_e` outcomes, probe `e` has `A^Z_e` deterministic
contingent policies. Version 1 deliberately fails closed above configurable
per-probe and total meta-action caps. Larger action/outcome spaces require a
separate optimization result; they must not silently approximate this exact
certificate.

## Claim boundary

The theorem is exact for the supplied finite hypothesis support, quotient
masses, direct-action losses, probe costs, and probe likelihoods. It does not
establish that these objects are physically correct, learned without leakage,
calibrated, stable under distribution shift, or safe for deployment. It is an
ex-ante expected-loss result and does not imply a per-outcome harm guarantee.
