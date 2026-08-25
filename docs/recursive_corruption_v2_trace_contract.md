# Trace contract

The v2 trace archive retains, for every fresh seed-domain, condition, method, and
forecast step:

- absolute error;
- update acceptance;
- exact fallback;
- materially harmful accepted update;
- fallback-reason code;
- corruption mask;
- declared reliability; and
- reported source age.

The archive is a deterministic ZIP of canonical NumPy arrays with fixed entry
metadata. Its stable SHA-256 and the exact `--check` regeneration path make the
reported time and recovery summaries auditable. Pickle is forbidden.
