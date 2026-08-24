# Metric support

- `materially_harmful_accepted_update_count` is defined for every updating arm
  and is `null` for `physical_baseline`.
- Gaussian NLL, nominal-90% coverage, and interval width are defined only for
  `recursive_gaussian` and `guarded_recursive`; they are `null` for deterministic
  arms.
- Exact fallback violations are defined for guarded arms and must be zero.

Undefined is never serialized or reported as measured zero.
