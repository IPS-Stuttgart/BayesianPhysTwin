# DEFORM public fidelity audit v1

This target-free audit clarifies two details of the public DEFORM benchmark
before any DLO2 source or evaluation data are opened.

## Metric and evaluation population

The paper's Table 1 reports average L1 loss over 500 prediction steps in units
of `10^-2 m`. The DLO1 and DLO2 references are therefore 10.1 mm and 9.7 mm
mean coordinate-wise L1, matching the Bayesian-PhysTwin metric.

The released evaluator samples 14 paths with replacement after a preceding
56-path training draw. Under seed 0 and a canonical sorted path population,
the evaluation draw contains only nine unique indices. The sealed v2 official
protocol therefore reports all 14 unique files and that canonical compatibility
draw, and requires the candidate to beat 9.7 mm under both views. Because the
upstream script does not specify the order returned by `glob.glob`, the
compatibility draw is not represented as an exact reconstruction of the paper
run's filesystem order.

## Training budget

The released script constructs 399 horizon-100 windows for each of 56 drawn
training trajectories. With batch size 32 and `drop_last=True`, that is 698
updates per epoch and 69,800 updates over 100 epochs.

The currently running DLO1 long-run v2 route is intentionally smaller: 6,400
horizon-50 updates over a 40-trajectory fit split. It is about 9.17% of the
released nominal optimizer-update count and 4.58% of the nominal unrolled
frame-step budget. This route follows separately recorded author stability
guidance and remains scientifically useful, especially if it reaches the
published accuracy region with much less compute, but it is not a
compute-matched reproduction of the public script.

No live artifact or gate is changed by this audit. If long-run v2 fails, the
correct conclusion is that the frozen 6,400-update route failed. It would not
falsify DEFORM under the public nominal training budget.

The checksummed source facts, arithmetic, and information boundary are recorded
in
`results/sota/deform_dlo2_official_eval_v2/reference_operator_audit.json`.
