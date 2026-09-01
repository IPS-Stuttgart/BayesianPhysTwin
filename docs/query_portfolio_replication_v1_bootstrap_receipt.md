# Slingshot bootstrap receipt

Before any Slingshot attempt ledger, prefix, future, or output directory was
created, the first operator launch exited while loading the frozen v2 custody
chain. The inherited runner expected its benchmark parent and terminal
policy-v1 roots under a host-specific `/home/fpfaff` path that does not exist on
the execution host. The observed terminal exception was `ValueError: exact
frozen parent root required`.

This is a zero-simulation, pre-science bootstrap failure. It does not authorize
changing seeds, worlds, policies, calibration, statistical gates, or output
semantics. The runtime amendment binds the same byte-verified roots under
explicit `/home/florianpfaff/source-only/dlolab-query-portfolio-replication-v1`
paths. The preserved bootstrap log remains outside the scientific output root.
