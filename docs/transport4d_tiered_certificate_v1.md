# Transport4D: tiered transport of learned physical corrections

## Question

A learned residual is not one indivisible object.  A change of simulator,
coordinates, object, material, contact, or action may preserve different parts
of the correction.  Transport4D asks:

> What is the strongest part of a learned physical correction that is justified
> for the registered target query before its held outcome is opened?

The answer is a fail-closed hierarchy rather than a binary transfer decision.

## Transport hierarchy

From strongest to weakest, the certificate considers:

1. **Exact coefficients.** Reuse the complete learned correction without target
   fitting.
2. **Query-identifiable effect.** Reuse an exact registered target-query effect,
   for example through a known equivariant push-forward or the interventional
   transport quotient. The latent physical cause need not be unique.
3. **Low-dimensional correction.** Retain a residual direction or basis and fit
   only a registered low-dimensional amplitude from permitted calibration data.
4. **Uncertainty only.** Reuse dependence or discrepancy structure without
   changing the deterministic target mean.
5. **Procedure only.** Reuse the architecture, fitting protocol, guard, and
   prior structure, but refit all target correction coefficients.
6. **Unsupported.** Transport nothing and return the exact registered fallback.

A tier is structurally eligible only if every source-frozen check passes and its
record is blind to the held target outcome. The first eligible tier in this
ordering is not automatically selected: deterministic mean tiers must also
identify a unique action inside the registered robust-regret budget.

## Exact query-conditional action test

Let the registered target query have baseline value `q0`. A mean-transport tier
supplies a transported effect `delta` and a deterministic Euclidean error radius
`rho`, so the unknown query lies in

\[
\mathcal Q = \{q_0 + \delta + e : \|e\|_2 \leq \rho\}.
\]

For finite actions with affine losses

\[
\ell_a(q)=w_a^\top q+c_a,
\]

the exact worst-case regret of action `a` on this ball is

\[
\overline R(a)
=\max_b\left[(w_a-w_b)^\top(q_0+\delta)
+c_a-c_b+\rho\|w_a-w_b\|_2\right].
\]

Transport4D selects a mean tier only when exactly one action minimizes this
quantity and its minimum is no larger than the registered tolerance. If a
stronger tier is structurally plausible but its error radius leaves the action
ambiguous, the policy descends to the next tier.

Uncertainty-only and procedure-only tiers may preserve a belief or a fitting
recipe, but they do not authorize a deterministic mean correction. Their action
is therefore the exact caller-owned fallback.

## Relationship to the interventional transport quotient

The transport quotient addresses a logically prior question. For an adequate
cause design `S`, residual `r`, and target map `T`, the compatible coefficient set
is

\[
S^\dagger r + \ker(S).
\]

The target effect is unique when `T ker(S) = {0}`. Such a fully identifiable
record supplies the query-identifiable-effect tier: its invariant target effect
is the Transport4D query correction and its reported perturbation bound can be
used as the query-error radius. Partial or nonidentifiable records fail that
tier and require a diagnostic intervention, a lower tier, or fallback.

Transport4D therefore does not duplicate cause-family adequacy or query
identifiability. It decides how much of their output may be deployed and whether
that amount is sufficient for the pending action.

## Public development evidence

The existing public DEFORM/PyElastica evidence exhibits a strict tier separation:

| Shift | Exact coefficients | One-scalar correction | Procedure refit |
|---|---:|---:|---:|
| Same DLO3, DEFORM to PyElastica | +2.985%, 8/8 wins | +2.250%, 6/8 wins | not the primary test |
| DLO3 coefficients to DLO4/DLO5 | -16.590%, 0/28 wins | not tested | +6.803%, 28/28 wins |

Thus, backend change on one physical object preserves exact correction value,
whereas object/operator change destroys it even though the same fitting
procedure succeeds after target-object refitting. The result motivates a
hierarchy; it does not confirm the newly designed selector because all listed
outcomes were already available during method development.

The same-mean Deform360 study independently motivates the uncertainty-only tier:
structured dependence improves logged decisions while predictive means and
coordinate marginals remain fixed. It is not yet a cross-domain uncertainty
transport result.

## Untouched public-data boundary

`transport4d_deform360_reserve_audit_v1.json` freezes a metadata-only reservation
of every Deform360 object namespace not already used or protected by the bound
v3 and v5 protocols. A deterministic hash rule assigns all remaining metadata
objects to calibration or confirmation before robot, tactile, image, geometry,
or target-outcome payloads are opened.

Later carrier qualification may mark an assigned object support-negative using
file identity and structure only. It may not replace that object or move it
between calibration and confirmation. Numeric confirmation access requires a
separate reviewed protocol after the tier models, thresholds, queries, losses,
and success criteria are frozen.

## Claim boundary

The exact action result is conditional on the supplied candidate checks,
transported query effects, error radii, affine loss portfolio, and regret
tolerance. The implementation does not prove a physical transformation correct,
learn an error radius, establish exchangeability, validate nonlinear rollout
closure, authorize target access, or certify deployment safety.
