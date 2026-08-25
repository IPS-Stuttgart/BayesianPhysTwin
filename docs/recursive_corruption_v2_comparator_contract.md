# Matched comparator contract

`guarded_last_residual` and `guarded_recursive` call the same guard with the same
availability, reliability, reported source age, normalized innovation,
observation variance, NIS threshold, update limit, residual limit, and exact
fallback representation. The deterministic arm accepts the measured residual;
the Gaussian arm applies its posterior gain. No method-specific cue or threshold
is permitted.
