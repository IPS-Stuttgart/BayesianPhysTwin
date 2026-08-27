# Native Support Task: Contact Works, Decision Gate Fails

Status: terminal source-design negative result. No Bayesian controller was
evaluated or promoted. The prior contact-free result and successful DEFORM
forecasts remain unchanged.

## Frozen Run

Implementation: `80dc5e2d3056c56e52895390e44ad83ab54a07c5`.
Native DLO-Lab: `c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`.
One CPU/float64 run generated all 12 actions in all six stiffness/support
worlds, followed by the registered contact and complete-state replay checks.
Every trajectory was sealed before the nine task goals were scored. There
were no replacements, retry, changed goals, or relaxed thresholds.

This uses only procedural geometry in the public native simulator. It is not
an official DLO-Lab benchmark reproduction or evidence from new recordings.

## Native Qualification

All 13 native checks passed. All six worlds made geometric support contact.
Maximum root error was exactly zero, maximum relative segment-length error
was 0.00124805 (0.124805%), and maximum support penetration was
4.80e-13 m, numerically negligible. Fixed-support positions were unchanged.
Two repeated action branches and one monolithic prefix-plus-action rollout
were byte-identical in positions and all 15 stored native memory fields.
Execution took 92.89 seconds. These checks establish the frozen interface and
replay behavior, not converged physical accuracy or real-material validity.

## Decision Sensitivity

At each fixed goal, compare the world-informed oracle with the best single
action chosen without knowing the world. Normalize their loss difference by
mean hold loss. The locked gate requires a difference of at least 10%, an
absolute difference of at least 25 mm^2, and at least two oracle actions in
at least three of the nine goals.

| Goal (x,z), m | Distinct oracle actions | World-information value / hold loss |
|---|---:|---:|
| (0.35,0.35) | 4 | 3.003% |
| (0.35,0.45) | 3 | 0.837% |
| (0.35,0.55) | 4 | 2.043% |
| (0.50,0.35) | 4 | 0.608% |
| (0.50,0.45) | 4 | 0.379% |
| (0.50,0.55) | 4 | 1.035% |
| (0.60,0.35) | 5 | 0.426% |
| (0.60,0.45) | 4 | 0.388% |
| (0.60,0.55) | 4 | 0.467% |

Zero of nine goals passed. Equal-goal mean loss was 0.08966764 m^2 for
hold, 0.07524794 m^2 for the best world-blind action, and 0.07448229 m^2 for
the oracle. Thus even full world knowledge improved this strong world-blind
comparator by only about 1.02% in aggregate.

Unlike the contact-free task, the best action now depends on the hidden
physical world. Nevertheless, this fixed action set, horizon, and loss offer
too little value of information to justify a larger Bayesian-control study
under the registered threshold. This does not reject Bayesian control or
contact inference generally. No method was fitted and no method-comparison
outcome is implied.

## Verification And Custody

The source runner independently recomputed all 648 scalar losses with an
explicit loop before sealing the decision result. A separate read-only
arithmetic script rechecked the same 648 losses, all nine goal decisions,
the aggregate decision, three position/memory replay equalities, canonical
record identities, and the sealed NPZ file digest. This is a second arithmetic
implementation, not an independent-person review.

New-plus-DLO-Lab focused tests passed 86/86. Expanded DEFORM, DEFT, native
restart, and harm-risk regression tests passed 439/439. Changed Python files
passed Ruff and focused MyPy; diff checks passed. The full repository suite
was not run for this isolated experiment.

Content identities:

- Lock: `de424971a14ed665712708bcd92f2365d6d6ba0d78f756762e1de70a91446d5f`.
- Native seal: `46c42cf734464198e56543ed167314bad2e3aae8a4bc4a3dec24205c0074481f`.
- Sensitivity: `932b62845673eb09b7f27a5dec34f006c1560762990f04d77e346cd2079ea9c2`.
- Native NPZ SHA-256: `7f810f5b0bf5ac3f8710dfe88961df5f8f5d4f6f5856a726a4ae15609d6950c6`.

Run root: `/home/fpfaff/source-only/dlolab-support-decision-source-v1/qualification-v1`.
The user-facing archive contains the complete numerical evidence, source
bundle, verification script, and checksums. Everything remains local/private.
No protected datasets, held-v8, DLO4/DLO5, official DLO3 evaluation, GPU work,
robot execution, public push, or main-branch merge was used.
