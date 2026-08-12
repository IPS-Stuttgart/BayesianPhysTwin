# Deform360 v6 per-case stdin isolation repair

The protected-main source execution at revision
`f8229074e01142e90357fd863beec0556229d9b4` retained a technical failure
after one complete physical manifest and before any source prediction seal.
The official PhysTwin Python 3.10 runtime had passed its bootstrap and had
executed the physical-prior stage successfully.

The next materializer invocation received an empty `--object-id`. The compact,
checksum-verified diagnostics establish the source-independent cause: commands
inside the shell roster loop inherited the roster file as standard input. A
child consumed the remaining stream, so the next shell `read` received only a
tail fragment.

Repair

- The three registered per-case dispatcher entry points receive `/dev/null` as
  standard input.
- Generic dispatcher calls preserve their previous stdin behavior, including
  the runner's existing Python heredocs.
- The dispatcher writes and validates an atomic, content-addressed activation
  marker. The compact execution receipt records whether that marker appeared.
- The archived scientific runner, source cohort, physical algorithm, selector,
  covariance, loss, prediction horizons, and all gates remain unchanged.

The retained run produced zero source prediction seals. It opened no
development suffix, confirmation payload, fresh target, or target outcome.
This repair therefore authorizes only a new protected-main source execution
after review; it does not authorize a scientific claim or target access.
