# DLO-Lab wrapping v9 calibration certificate

The already-fixed v8 0.975 posterior guard harmed 2 of 144 independent
public-simulator worlds under the registered event: mean reward over 4096
sensor draws more than `0.002` below the exact fixed-action fallback. The
observed rate is `1.389%`; the exact one-sided 95% Clopper-Pearson upper bound
is `4.307%`.

That bound is below the v9 risk budget of `5%`, so the certify-or-fallback
wrapper admits the unchanged action policy for one disjoint replication. The
guard's v8 mean gain over fallback was `0.005042` with paired 95% bootstrap CI
`[0.003726, 0.006409]`.

This is a post-open calibration certificate, not v9 evidence. The 5% budget was
chosen and frozen for v9 after v8 opened, while the candidate threshold itself
was registered before v8 outcomes and was not retuned. V8 remains a strict
source-gate failure under its original zero-harm and value-retention criteria.

The certificate assumes exchangeable worlds from the registered simulator
stress distribution and concerns the sensor-averaged simulator estimand. It is
not evidence of per-trajectory safety, real-robot safety, official benchmark
superiority, or point-prediction state of the art.
