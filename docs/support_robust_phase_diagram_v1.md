# Exact support-robust Act–Sense–Fallback phase diagram v1

## Question

For which amounts of unrepresented physical probability mass should a physical
twin act immediately, acquire a task-directed observation, switch to a more
conservative probe, or reproduce a caller-owned fallback?

A few evaluations at hand-picked support-miss values do not answer this
question. The phase-diagram certificate computes **all** transition values on a
registered interval without an epsilon grid.

## Represented-support certificate

Let `p` and `b` denote complete plans. A plan is either a direct terminal action
or a deterministic probe together with its frozen outcome-to-action map. The
existing quotient certificate returns the exact represented-support pairwise
gap

\[
D_0(p,b)
=\sup_{q\in\mathcal Q_0}
\mathbb E_q[L(p,H)-L(b,H)],
\]

where \(\mathcal Q_0\) contains every complete belief with the registered
quotient masses and prior support.

## Bounded missing support

Suppose at most \(\epsilon\) probability mass may instead lie on unknown
physics. The unknown complete plan-loss vector belongs to a declared
axis-aligned box

\[
\underline L_p\le L_p^{\rm miss}\le\overline L_p.
\]

For distinct plans, define

\[
M(p,b)=\overline L_p-\underline L_b.
\]

Because the adversary may use any missing mass between zero and
\(\epsilon\), the exact contaminated-support gap is

\[
\boxed{
D_\epsilon(p,b)
=D_0(p,b)
+\epsilon\,[M(p,b)-D_0(p,b)]_+
}.
\]

The diagonal is set to zero: a plan compared with itself has exactly zero loss
gap. The formula is exact for the declared ambiguity class. It is not a bound
obtained by a union inequality or an epsilon discretization.

## Piecewise-linear decision geometry

For one plan,

\[
R_p(\epsilon)=\max_b D_\epsilon(p,b)
\]

is the upper envelope of finitely many affine nondecreasing functions. The best
available plan has

\[
R_*(\epsilon)=\min_p R_p(\epsilon).
\]

The certified output is the deterministic lowest-index minimax plan whenever
\(R_*(\epsilon)\le\tau\), for registered tolerance \(\tau\). Otherwise it is
the exact caller-owned fallback.

Every possible transition occurs at one of three finite event types:

1. two benchmark lines cross inside one plan-regret envelope;
2. two plan-regret envelope segments cross; or
3. one plan-regret segment crosses the tolerance.

The implementation first constructs each affine upper envelope, then enumerates
those finite intersections. It returns:

- the sorted exact breakpoint vector;
- the decision at every breakpoint;
- the constant decision on every open interval between breakpoints;
- the active worst-case benchmark for each decision; and
- the largest admissible support-miss probability for every plan.

No epsilon grid is used. The implementation fails closed when the registered
plan or breakpoint cap is exceeded.

## Unknown-loss interfaces

The caller may provide a complete plan-level box directly. Alternatively, the
caller may provide terminal-action loss bounds. Direct plans inherit their
action bounds. A sensing plan adds its probe cost and permits unknown physics to
produce any registered outcome, giving

\[
\underline L_p=c_s+\min_o\underline L_{a_p(o)},\qquad
\overline L_p=c_s+\max_o\overline L_{a_p(o)}.
\]

This induced box deliberately does not assume that the represented probe model
remains correct off support.

## Why this matters for the paper

The phase diagram turns “robustness to model incompleteness” into an observable
scientific object. It supplies exact critical values for:

- when direct action ceases to be admissible;
- when a cheap but misspecification-sensitive probe is overtaken by a safer
  probe;
- how much missing support each complete plan can tolerate; and
- when every learned plan fails and fallback becomes mandatory.

This makes the Act–Sense–Fallback claim falsifiable and removes arbitrary
hand-selection of support-miss values from the main result.

## Complexity

Let \(P\) be the number of enumerated complete plans. There are \(P^2\)
pairwise affine gaps. The implementation constructs one upper hull per plan and
intersects only hull segments. It remains a finite exact method, but plan
enumeration can be exponential in probe outcome count. `max_plan_count`,
`maximum_phase_plan_count`, and `maximum_breakpoint_count` are therefore explicit
fail-closed resource contracts.

## Claim boundary

The phase diagram is conditional on the finite represented hypotheses, quotient,
loss matrix, deterministic registered probe maps, costs, tolerance, and declared
unknown plan-loss box. It does not estimate the actual support-miss probability,
validate the box or probe physics, prove reset semantics, establish target
transport, calibrate uncertainty, authorize online execution, or certify
safety.
