# DLO-Lab Coiling Off-Grid Source v2 Result

The source screen completed technically but failed its frozen transfer gate.
All twelve off-grid state/material worlds executed, sealed, and passed every
native qualification check. The source bank and leave-one-world-out analysis
were therefore complete; this is a scientific negative, not a runtime failure.

The best fixed source action was `counterclockwise_medium`, with mean native
reward `0.1117041575`, compared with `0.0913311597` for the prefix hold. The
per-world oracle reached only `0.1133103543`, leaving `0.0016061968` reward of
oracle headroom. That is below the frozen adjusted-headroom gate and shows that
the current action bank offers little transferable choice beyond its best fixed
member.

The guarded policy fell back exactly to the fold-specific fixed action in nine
worlds. It admitted `clockwise_fast` in three worlds. Those decisions produced
gains of `-0.0011920915`, `-0.0063806487`, and `+0.0024087314`, respectively.
The complete cross-fitted mean gain was therefore `-0.0004303341`. Mean
observation-draw harm probability was `0.0833333`, and the worst world was
`1.0`; both exceeded their frozen limits.

No source threshold is relaxed after seeing these outcomes. The coiling action
bank, prefix observation, and guarded transfer rule are closed as a candidate
for prospective promotion. No prospective worlds were selected or run, no
protected data were read, and no retry is authorized. This result supports the
broader query-conditional conclusion: physically valid simulator variation and
Bayesian abstention are not enough when the registered action bank has too
little oracle headroom and the source mapping does not transfer safely.

Compact evidence is in
`results/source/dlolab_coiling_offgrid_source_v2/summary.json`. The sealed run
remains at `/home/fpfaff/source-only/dlolab-coiling-offgrid-source-v2`.
