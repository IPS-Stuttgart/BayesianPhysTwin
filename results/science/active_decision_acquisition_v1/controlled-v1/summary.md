# Controlled active decision-acquisition result

Decision: `controlled-active-decision-acquisition-passed`
Result ID: `3145f03d305760e842807896b6aeff7fe50fa803a46d43e3b84bb40b42dfebd0`

| Method | Worst-case probe cost | Uniform expected cost |
| --- | ---: | ---: |
| Exact adaptive decision certificate | 2.000 | 1.333 |
| Greedy hypothesis entropy | 4.000 | 3.333 |
| Global decision-identifying set | 2.000 | n/a |
| Full-state-identifying set | 6.000 | n/a |

The adaptive policy spends one probe on the 16/24 hypothesis branch whose decision is immediately identified and a second probe only on the remaining branch. Complete state identification requires all six registered probes.

Removing `decision-probe-1` leaves an observationally indistinguishable pair with opposing optimal actions; the exact policy reports the task as infeasible and therefore fails closed.

This is controlled finite-hypothesis evidence, not a real sensor result.
