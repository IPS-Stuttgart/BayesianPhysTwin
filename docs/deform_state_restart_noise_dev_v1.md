# DEFORM State Restart: Simulated Measurement Robustness

The parent opened-data experiment at `b85b8162` found a roughly 10% average
hidden-point gain from propagating the incumbent's remaining sparse pose and
velocity residual through DEFORM. That result is already opened and is preserved
unchanged. This is a separately frozen, explicitly post-result diagnostic, not
confirmation on a fresh cohort.

Keep the parent's roster, prefix/forecast window, eight-point measurement budget,
interpolation, native checkpoint, and disjoint hidden identities. Compare both
previously declared pose/velocity gains (1 and 0.25), without reselecting them,
against unchanged incumbent and matched full-gain readout persistence.

For each of 16 fixed random seeds, add either 1 mm independent coordinate noise,
or that identical noise plus one 5 mm-standard-deviation 3D translation shared
across all eight observations of each trajectory. The translation is a simulated
measurement nuisance, not a known displacement of the physical object. No claim
is made that these scales describe an actual deployed sensor.

Generate all conditions/arms before scoring. Average metrics over the 16 noise
realizations within each trajectory, then over the same 13 non-design
trajectories. Bootstrap whole trajectories, not the correlated noise repetitions
or coordinates. Report all four arms in both conditions, including regressions.
The parent numerical-parity and exact-fallback controls remain mandatory. Neither
the original successful DEFORM result nor the parent state-restart result changes.
