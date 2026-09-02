# Prospective portfolio replication v5 recovery

Slingshot v4 sealed all calibration data and all 320 evaluation prefixes and
candidates, then terminated at the evaluation decision barrier before any
evaluation future was executed. Two guard helpers imported from the frozen v2
implementation still read v2's 288-world module global even though the v3/v4
modules had been configured for the registered 320-world denominator.

V5 configures the transitive v2, v3, and v4 modules consistently and freezes a
source-only shape preflight for those inherited helpers. It uses new calibration,
evaluation, and sensor seeds and reuses no v4 world. No scientific policy,
calibration rule, estimand, threshold, or familywise error allocation changes.
