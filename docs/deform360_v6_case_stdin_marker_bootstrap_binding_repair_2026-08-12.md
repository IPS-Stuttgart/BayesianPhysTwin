# Deform360 v6 stdin-marker bootstrap binding repair

The protected-main source execution at revision
`2350d1bde3b1a7b20a3602e7fd08f01aecab882b` retained a technical failure
before prediction generation. Both dual-runtime contract jobs passed and the
isolated CUDA environments built successfully, but the first generic
dispatcher probe exited because its required
`BPT_CASE_STDIN_ISOLATION_MARKER` pathname had not yet been bound.

Repair

- The runtime bootstrap binds the already-registered marker pathname before
  the first dispatcher call.
- The generic bootstrap probe still preserves its Python heredoc on standard
  input and does not create the per-case activation marker.
- Registered per-case dispatcher calls remain solely responsible for creating
  and validating the content-addressed marker and for redirecting their own
  standard input to `/dev/null`.
- The dispatcher, archived scientific runner, source cohort, physical
  algorithm, selector, covariance, loss, horizons, and all gates are unchanged.

The retained execution produced zero physical manifests and zero source
prediction seals. It opened no development suffix, confirmation payload, fresh
target, or target outcome. This repair authorizes only one new protected-main
source execution after review; it does not authorize a scientific claim or
target access.
