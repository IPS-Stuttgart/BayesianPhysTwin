# Deform360 dependence robustness v7 — retained result

## Decision

This retrospective attribution test gives a **mixed result**.

The full low-rank covariance retains a statistically supported advantage when all
three covariance arms receive the same fixed deployment budget across the five-query
portfolio. It does **not** retain general superiority after source-only
leave-one-object-out dispersion recalibration. The result therefore supports a
narrow query-allocation claim under the frozen original calibration, not an
unrestricted claim that dependence structure itself is always better.

## Execution and custody

- Successful workflow: `33750898386`
- Trigger revision: `c4dc1aa738d47357f415e083ba88dd7786c241a9`
- Runner label: `gpuserver4090`; machine: `workstation1`
- Dataset: `/mnt/seagate10tb/florianpfaff/datasets/deform360`
- Complete capture: 92 objects and 460 object-query cases
- Exact v6 scientific projection reproduced: `true`
- New measurements, pixels, geometry, and point clouds opened: none
- Workflow artifact: `9891515892`, SHA-256 `b2d05f3b2da97f61399d20e81096087555ee984cd9fa6b2d72ac479b44da5dd0`
- Internal result digest: `68fd84174a536f434c56febffdd34a36fabb90fb6b4872863b7f6d77b79ea39e`

The first launch (`33750165627`) failed before reading data because NumPy was
imported before the isolated environment existed. The technical retry changed only
the workflow ordering; the frozen analysis payload and scientific request were
unchanged.

## Result that survives: equal-budget query portfolio

At 40% matched coverage, every arm accepts exactly the same number of target cases:

| Arm | Decision loss | Harmful accepted / all |
| --- | ---: | ---: |
| Full low-rank | 0.117690 | 5.769% |
| Diagonal marginal-matched | 0.120326 | 6.033% |
| Scrambled marginal-matched | 0.120841 | 6.084% |

Full minus diagonal decision loss is
`-0.002636 [-0.004714, -0.000772]`;
full minus scrambled is
`-0.003151 [-0.005068, -0.001282]`.
Both object-bootstrap intervals exclude zero. Relative to the comparators, these
are reductions of 2.19% and 2.61%, respectively.

Across the complete 0–100% fixed coverage grid, decision-loss AURC is
`0.156794` for full, `0.158834`
for diagonal, and `0.159032` for scrambled.
The paired differences are
`-0.002041 [-0.002866, -0.001267]` and
`-0.002239 [-0.002989, -0.001525]`. This is the strongest
retained evidence: the original full covariance ranks a finite portfolio of
query cases more usefully under a fixed fallback budget.

## Attribution that fails: source-only temperature controls

A single leave-one-object-out source scale per covariance arm removes strict
superiority. Full versus diagonal decision loss is
`0.001256 [-0.001123, 0.003771]` and full
versus scrambled is
`0.001548 [-0.000886, 0.004019]`; both
intervals include zero. Full query NLL is significantly worse by
`0.690223 [0.116014, 1.413507]` versus diagonal
and `0.660590 [0.138080, 1.305377]` versus
scrambled.

With one source-only leave-one-object-out scale per query and arm, the control
reverses the ordering. Full is worse than diagonal by
`0.002579 [0.000700, 0.004563]` in decision
loss and `0.001776 [0.000760, 0.002856]` in
Brier score; it is worse than scrambled by
`0.003295 [0.001136, 0.005572]` and
`0.001884 [0.000749, 0.003059]`. All four
intervals exclude zero.

The required source scales explain why: the global median multiplier is about
`1.344` for full, but
`8.035` for diagonal and
`8.802` for scrambled.
Thus much of the original proper-score advantage came from dispersion magnitude,
not uniquely from the detailed dependence topology.

An independently fitted source scale for every object-query-arm forces numerical
parity by construction: maximum Brier difference
`5.551e-17`
and maximum decision-loss difference
`0.000e+00`.

## Additional falsification

The assumption that equal-coverage rankings would be identical within each query
was false: the retained gate is `false`, and 23796
mask entries differ across arms over the registered grid. At 40% within-query
coverage, full is not significantly better than either comparator. The portfolio
result should therefore be described as cross-query budget allocation, not as a
universal within-query ranking result.

## Defensible manuscript wording

> With frozen original calibration and identical predictive means and coordinate
> marginals, structured covariance improves fixed-budget allocation across a
> registered five-query portfolio. The advantage does not survive flexible
> source-only variance-temperature controls, so we do not claim universal
> dependence superiority or calibrated uncertainty.

This evidence is retrospective and uses already opened public targets. It does
not authorize fresh confirmation, robot-control safety, calibrated raw-field
uncertainty, or a general Bayesian-superiority claim.
