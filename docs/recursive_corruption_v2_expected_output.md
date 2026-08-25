# Recursive-corruption v2 output contract

A complete result directory contains:

- `result.json`: complete seed-domain, condition, method, configuration, and
  access-boundary record;
- `records.csv`: long-form scalar metrics;
- `traces.npz`: deterministic per-time-step trace archive;
- `analysis.json`: primary, secondary, co-equal, condition, and time summaries;
- `metric-support.json`: methods for which each endpoint is defined;
- `condition-summary.csv` and `endpoint-summary.csv`;
- `time-summary.csv`: corruption-aligned recovery data;
- `result-note.md`; and
- `analysis-manifest.json`: SHA-256 bindings for exact regeneration.

`python -m bayesian_phystwin.recursive_corruption_benchmark_v2 --check` must
regenerate every product byte-for-byte. Any mismatch blocks evidence retention.
