# Active-probe Native Wrapping Source Screen

This is a new bounded public-simulator source study, frozen before any new native
trajectory. It tests dual control rather than another DEFORM forecast correction.
The successful paired DEFORM update, the prior passive wrapping result, and every
failed gate remain unchanged. Work is local/private and uses no new recording,
real robot, GPU, held-v8, DLO4/DLO5, official DLO3 evaluation, PokeFlex
continuation, or fresh/reserved Deform360 data.

The first v1 invocation stopped before lock publication or native initialization
because its shell omitted the registered CPU/OSMesa environment. It produced zero
trajectory and no scientific payload. Its original receipt accidentally recorded
an incorrect revision; both that immutable receipt and a correction receipt are
bound in the repository. The separately versioned v1.1 runner preflights runtime
and source identity before creating a new output root. The old root is terminal and
cannot be reused. This is a bootstrap correction, not a retry of empirical science.

## Scientific Question

Can a reversible, reward-blind probe make uncertain material belief useful for a
later native wrapping decision? The probe is selected only by expected material
mutual information from causal prefix trajectories. Future wrapping reward does
not enter probe selection. Both the selected probe and a matched null probe are
then evaluated with the same continuation actions, material prior, observation
model, and decision rules.

This separates three quantities:

1. whether a safe probe creates distinguishable material responses;
2. whether those responses improve posterior action selection;
3. whether any apparent gain exceeds the matched probe-independent fixed policy.

## Frozen Native Design

The public DLO-Lab `Train_Env_Wrapping` task remains unchanged: the same 50-node
extensible loop, two robots, three posts, solver, contact, reward, and native
failure rules are used. Existing material-randomization hooks are fixed to the
same nine equally weighted stretching/bending settings as the passive study.
These are simulator settings, not identified material constants.

Four geometry-defined probes share the same initial lift and approach. The null
probe holds. The three active probes apply a 40 mm symmetric tension, symmetric
compression, or axial shear excursion. Every probe returns to the same waypoint,
holds for one macro, and then reaches the same pre-continuation tool pose. No
gripper displacement exceeds 100 mm per macro. The native reward, action bank,
and continuation waypoints are not used to choose the probe.

The probe bank runs first across nine material worlds, with three nominal repeats.
Only nodes 0/8/25/41/49 at frames 399/599/799/999/1199 enter selection. A uniform
nine-particle model assumes independent 2 mm coordinate noise plus one shared
5 mm translation over the prefix. The selected probe maximizes expected mutual
information; ties use the lowest index. It advances only if it is non-null,
improves mutual information by at least 0.05 nat, and improves material
classification by at least 0.05 over null.

If and only if that gate passes, two complete continuation banks run: null and
selected. Each uses the prior eight unique wrapping motions plus the unchanged
duplicate, eleven material/repeat batches, and the unchanged final winding reward.
The two conditions use common Monte Carlo noise draws. The previous passive result
is a frozen descriptive comparator and is not a probe-selection input.

## Advancement Gate

All native/repeat/duplicate/fixed-post/attachment/segment checks must pass. The
active condition must also satisfy every decision check:

- best fixed beats hold by at least 0.05 reward;
- adjusted oracle headroom is at least 0.02 and at least two oracle actions occur;
- adjusted Bayesian gain over active best-fixed is at least 0.02 and at least 5%
  of its remaining deficit from reward one;
- Bayesian reward is noninferior to MAP within the fixed 0.002 numerical margin;
- active Bayesian reward beats null Bayesian reward by at least 0.005;
- Bayesian gain over fixed improves by at least 0.005 versus null;
- at least five of nine material worlds improve under active Bayesian selection.

Failure stops the study. There is one registered attempt, no replacement, retry,
threshold relaxation, alternate probe rescue, or post-result action-bank change.
Fallback retains the exact prior passive wrapping and DEFORM evidence.

## Claim Boundary

A pass would be finite-prior native source evidence for active Bayesian probing,
not independent generalization, real perception calibration, real counterfactual
evidence, a new Bayesian-decision theorem, official benchmark parity, or SOTA.
All evaluation worlds lie on the exact model support and observation noise is
synthetic. A pass would justify a separately locked evaluation, not authorize one.
