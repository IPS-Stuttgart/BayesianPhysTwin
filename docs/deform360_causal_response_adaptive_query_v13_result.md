# Deform360 Causal-Response Adaptive Query V13 Result

## Decision

**The frozen post-open source carrier gate passes exactly.**

| Target-free disposition | Count |
| --- | ---: |
| Complete 16-query schedule | 6 / 8 |
| Strict 3+3 admission | 2 / 8 |
| Inflated 2+2 admission | 4 / 8 |
| Exact abstention | 2 / 8 |
| Technical failure | 0 / 8 |

The registered gate required at least six total admissions, at least two strict
admissions, and no technical failures. V13 satisfies all three conditions.

No tactile stream, tracker output, state innovation, state update, future
identity, future object observation, future metric, V1 target, or held-v8
artifact or process was read.

## Frozen Evidence

- V12 closed result commit:
  `cf2b532e01a3c92d1761f5bdea36a7c026e2c3b8`
- V13 implementation commit:
  `bdde57f790fdc6c41255d6968e1387e97c381062`
- V13 pre-disposition protocol commit:
  `ade5de6bac49551b994b7e6a0989a4da805a67bb`
- Protocol canonical SHA-256:
  `2cff14d209a314c618b1381800cb179ce0d8632c9f66338fb79ee75b6571862f`
- Result canonical SHA-256:
  `63a9ffde07de4e0378605b8574a8d4a8acfe6a382c65df2be185c47f6276239c`
- Result file SHA-256:
  `851a034984e9661a1e18097f514bc780967d5254afbd6ce1cf20fc1b086ad1b0`

The complete query artifacts are under
`results/sota/diagnostics/deform360_causal_response_adaptive_query_v13_source/`.

## Interpretation

Adaptive complete-camera selection removes V12's four exact-panel technical
failures. It also recovers one previously unsupported case with strict support
and three more through the declared two-view fallback. The resulting carrier
coverage rises from 2/8 under V12 to 6/8 under V13.

This is not yet evidence that the observations are accurate or that a Bayesian
state update improves prediction. Four of six admissions rely on two views per
independent panel. Those rows retain a fourfold local-covariance inflation and a
separate 5 mm shared-bias nuisance; they must not be interpreted as equivalent
to strict 3+3 observations.

Two cases still have zero action-supported identities satisfying even 2+2
association support. They remain exact abstentions. The result does not
authorize weaker support, fewer queries, single-view updates, or replacement
cases.

## Authorized Next Step

The carrier gate authorizes a separately frozen source-only study on the six
admitted cases:

1. infer the earliest tactile/contact event from the permitted prefix;
2. run a preregistered tracker provider only on the sealed query identities;
3. require provider competence before constructing an update;
4. use nuisance-aware, covariance-inflated Bayesian state inference;
5. compare every candidate against an exact unchanged physical baseline;
6. retain exact fallback whenever admission or regret gates fail.

That study remains post-open development. Fresh-object evaluation is still
prohibited until an independent held-v8 all-attempt hash-only exclusion
manifest exists.
