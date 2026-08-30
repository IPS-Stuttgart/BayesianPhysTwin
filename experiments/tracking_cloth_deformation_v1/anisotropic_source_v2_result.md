# Tracking Cloth anisotropic active-probe source feasibility v2 — result

Status: **source-only negative; do not promote to target evaluation**.

GitHub Actions run: `33321314927`

Trigger revision: `ba229e7a0f7c59e01dc5f2793e0af480de449e95`

Artifact: `tracking-cloth-anisotropic-source-v2-33321314927`

Artifact digest: `sha256:5934155f3987aec0d1d0739cbfe3e980cbf455d7aefe6ead75789cc2fe114b5f`

## Frozen source-only question

Would a richer 55-member anisotropic spring-mesh bank create nontrivial
query-specific K=1 probe decisions under the existing leave-one-material-out
Shake-to-Twist input protocol, without reading any Twist free-marker outcome?

The predeclared follow-up gate required at least 2/8 material-size specimens on
which task-directed and parameter-information selection chose different first
probes.

## Result

- task-directed K=1 choice: `fast_hanger` for 8/8 specimens;
- parameter-information K=1 choice: `fast_hanger` for 8/8 specimens;
- fixed-order K=1 choice: `fast_hands` for 8/8 specimens;
- task-vs-parameter disagreement: `0/8` (`0%`);
- minimum useful divergence gate: **failed**;
- Twist free-marker outcomes read by this run: **no**;
- target scoring authorized: **no**;
- paper claim authorized: **no**.

## Scientific decision

Do not tune this bank against the already exposed Twist outcomes and do not open
a new Twist evaluation for this model family.  The negative result indicates
that, for this logged action roster and these finite spring-model families,
`fast_hanger` is effectively dominant for both generic parameter information and
the registered averaged Twist-query spread objective.

A future query-directed acquisition claim should therefore use a separately
registered untouched query/action roster where candidate probes provide
complementary rather than uniformly ordered information.  The 56 unused
collision recordings remain potential data for a separately frozen experiment;
they were not numerically opened by this source-only run.
