# Finite-group calibration of decision-capability atlases

## Purpose

The affine decision-capability atlas is exact relative to a registered finite
physical support. That exactness does not imply that the support covers a future
real object. `decision_capability_calibration_v1` adds a separate statistical
correction from one fixed score per independent calibration group.

The resulting contract distinguishes three layers:

1. latent physical ambiguity inside the registered quotient;
2. uncertainty in the downstream task objective;
3. observed undercoverage of the registered physical support on independent
   calibration groups.

## Exact continuous-task group score

For one proposed action `a` and benchmark action `b`, let the registered
pairwise envelope over task parameter `theta` be

```text
M_ab(theta) = max_r (v_r + u_r @ theta).
```

Each affine witness `r` corresponds to one positive-prior complete physical
hypothesis per posterior-supported quotient class. Let the realized pairwise
loss gap on a calibration case be

```text
D_ab(theta) = e_ab + d_ab @ theta.
```

The undercoverage score for this action pair over a registered task box is

```text
sup_theta [D_ab(theta) - M_ab(theta)]
  = sup_theta min_r [(e_ab-v_r) + (d_ab-u_r) @ theta].
```

`maximize_affine_lower_envelope_on_box` solves this exactly for a
low-dimensional task family by enumerating vertices of the equivalent linear
program. It does not sample a task grid. A caller-supplied active-set cap causes
fail-closed refusal before combinatorial enumeration becomes too large.

`affine_box_pairwise_undercoverage_score` maximizes additionally over every
benchmark action represented by one action region. The experiment owner must
then take the maximum over every decision case and proposed action within one
independent physical object or trajectory. Only that single group-level maximum
may enter calibration.

## Finite-group correction

For exchangeable group scores `S_1, ..., S_n`, register `alpha` and compute

```text
k = ceil((n + 1) * (1 - alpha)).
```

When `k <= n`, the correction is the `k`-th order statistic. By default it is
clipped below at zero so calibration can shrink but cannot enlarge the
model-side atlas. If `k > n`, the implementation refuses to return a finite
correction.

The guarantee is marginal over one future exchangeable group. It is not a
pointwise, conditional, distribution-shift, or deployment-safety guarantee.

## Corrected capability geometry

An uncalibrated action region has half-spaces

```text
normal[r] @ theta <= offset[r].
```

For nonnegative correction `q`, `statistically_corrected_halfspaces` returns

```text
normal[r] @ theta <= offset[r] - q.
```

Equivalently, every registered pairwise loss-gap envelope is enlarged by `q`
before comparing it with the regret tolerance. Normals, benchmark actions, and
physical witness hypotheses remain unchanged, so every exclusion retains its
original explanation.

The correction composes additively with objective uncertainty. If a task box
with half-width `w` contributes support value

```text
abs(normal[r]) @ w,
```

then the combined robust-center condition is

```text
normal[r] @ center
  + abs(normal[r]) @ w
  + q
  <= offset[r].
```

The order of objective-set erosion and statistical erosion is therefore
irrelevant.

## Intended information order

1. Freeze the physical model, quotient, action set, affine loss family, task
   domain, scoring rule, and complete-group split.
2. Construct model-side pairwise envelopes without target outcomes.
3. Open calibration outcomes and produce exactly one maximum score per complete
   calibration object or trajectory.
4. Compute the fixed order-statistic correction once.
5. Apply the correction to untouched target-group capability regions.
6. Report capability volume, fallback rate, realized regret-limit violations,
   and the complete-group violation count.

Frames, windows, task points, and actions inside one physical group are not
independent calibration observations.

## Claim boundary

The correction is finite-sample marginal only under exchangeability of complete
calibration and target groups and a fixed pre-calibration model, quotient, task
domain, action set, loss family, and score. It does not validate those
ingredients, provide conditional or pointwise safety, cover distribution shift,
or authorize deployment.
