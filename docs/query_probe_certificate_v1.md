# Exact decision-directed probing under quotient ambiguity

## Setting

Let `H={1,...,m}` be a finite registered physical hypothesis set with prior
weights `p`.  Positive prior mass defines the admissible support.  An observation
determines only posterior masses `lambda_c` over a registered partition
`C_1,...,C_K`; it does not determine the conditional distribution inside each
class.  The compatible complete-belief set is

\[
\mathcal Q(\lambda)=\left\{q\ge 0:
q_i=0\text{ if }p_i=0,\quad
\sum_{i\in C_c}q_i=\lambda_c\right\}.
\]

For finite actions `a in A`, let `L(i,a)` be the registered loss under hypothesis
`i`.  A finite probe `e` has outcomes `z in Z_e` and registered likelihood
`O_e(i,z)`.  Before observing the probe outcome, a contingent policy
`delta: Z_e -> A` is fixed.

The expected post-probe action loss under complete belief `q` is

\[
R_e(q,\delta)=c_e+\sum_i q_i\sum_z O_e(i,z)L(i,\delta(z)),
\]

where `c_e` is a deterministic probe cost expressed in the same loss units.

## Theorem: exact robust regret of a probe policy

For any contingent policy `delta`, its worst-case ex-ante regret over every
compatible complete belief is

\[
\overline{\operatorname{Reg}}_e(\delta)
=
\max_{\delta'}
\sum_c \lambda_c
\max_{i\in C_c:p_i>0}
\sum_z O_e(i,z)
\bigl[L(i,\delta(z))-L(i,\delta'(z))\bigr].
\]

Consequently, the exact minimax probe policy is obtained by minimizing the
right-hand side over the finite policy set `A^{|Z_e|}`.

### Proof

For a fixed complete belief,

\[
R_e(q,\delta)-\min_{\delta'}R_e(q,\delta')
=
\max_{\delta'}
\sum_i q_i\sum_z O_e(i,z)
\bigl[L(i,\delta(z))-L(i,\delta'(z))\bigr].
\]

The comparator set is finite, so supremum and maximum commute.  For fixed
`delta,delta'`, the objective is linear in `q`.  The constraints on `q` separate
by quotient class.  Within each class, its fixed mass can concentrate on any
prior-supported maximizing hypothesis.  The support function is therefore
exactly the displayed class-wise weighted maximum.  Minimization over the
finite candidate-policy set completes the result.

## Corollaries

### Direct actions are the one-outcome special case

When the probe has one certain outcome, every contingent policy is one direct
action and the theorem reduces exactly to `query_decision_certificate_v1`.

### A state can remain unidentified while the probe policy is identified

Consider two hypotheses in one quotient class and two opposing actions.  No
direct action has zero robust regret.  A binary probe that perfectly separates
the hypotheses has contingent policy `(action_0, action_1)` with zero robust
regret.  The probe resolves the decision without requiring a unique
pre-probe physical state.

### An uninformative probe can still randomize

If probe outcomes have the same distribution under every hypothesis, they carry
no state information.  A deterministic outcome-contingent policy can nevertheless
use the outcome as exogenous randomization and reduce deterministic minimax
regret.  Empirical claims of decision-directed information value must therefore
include an uninformative or likelihood-scrambled probe control, rather than
comparing only with deterministic no-probe actions.

## Act, probe, or fallback

The implemented router uses separately registered tolerances:

1. **Act:** if the direct certificate admits an action at its regret tolerance,
   execute the deterministic minimax direct action.
2. **Probe:** otherwise, choose the lowest-index probe minimizing
   `probe_cost + exact minimax probe regret`, provided that value does not exceed
   a registered limit; after the outcome, execute its frozen contingent action.
3. **Fallback:** otherwise, return the caller's registered fallback action.

The tie-breaking and fallback action are part of the executable contract.

## Computational boundary

For `A` actions and `Z` outcomes, exhaustive contingent-policy enumeration costs
`A^Z`.  Version 1 deliberately fails closed above a configurable policy-count
cap.  Larger action/outcome spaces require a separate optimization result; they
must not silently approximate this exact certificate.

## Claim boundary

The theorem is exact for the supplied finite hypothesis support, quotient
masses, action losses, and probe likelihood.  It does not establish that these
objects are physically correct, learned without leakage, calibrated, stable
under distribution shift, or safe for deployment.  It is an ex-ante expected
regret result and does not imply a per-outcome harm guarantee.
