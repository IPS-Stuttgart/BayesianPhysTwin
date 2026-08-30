# DLO-Lab Coiling Development v1 Result

The frozen v1 attempt is a terminal pre-science implementation failure. It
created its write-once lock and first world claim, constructed the unchanged
public coiling scene, and stopped during reset before the first native scene
step. No trajectory, reward, source bank, value statistic, or development gate
was generated.

The defect is exact: the new material-binding assertion required the solver
field shape `(3, 8)`, copied from the three-rod wiring environment. Coiling has
one rod, so its field has one rod row. The native setter had not yet executed
any rollout step when the mismatched assertion failed. This says nothing about
coiling competence, action value, uncertainty, or transfer.

The original root remains terminal with `retry_authorized=false`. One separate
v1.1 implementation replacement may correct only this environment-specific
field shape. It must bind this failure, use a fresh output root, preserve the
action bytes, material worlds, observation policy, reward, noise model, gates,
and no-retry rule exactly, and still write each claim before native
initialization. Any other change requires a new development question.

Compact evidence is in
`results/source/dlolab_coiling_query_competence_development_v1/summary.json`.
The parent lock ID is
`4e7c5001ba5dcdbb3e49d19fa8b816343e89c272ef180d10a71c49dde9f1f5e6`;
the terminal failure ID is
`eb815bffa5914090ddad03ecc919a9c79fa4c6189f79bdce35714737f03f45f3`.
