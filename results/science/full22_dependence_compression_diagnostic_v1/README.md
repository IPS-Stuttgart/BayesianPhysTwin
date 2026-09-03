# Full-22 dependence and compression diagnostic

The target-free diagnostic is complete and negative for the proposed fused
headline. It analyzed 3,638,553 sealed per-point 3D covariance blocks from all
22 released PhysTwin sessions without extracting future truth, prediction means,
or prefix arrays.

The frozen scored covariance has equal-case mean total correlation
`0.0015531253` nats (`0.0022406862` bits) per 3D point, with a case-bootstrap
95% interval `[0.0010322945, 0.0020976714]`. No case reaches the predeclared
`0.01`-nat floor. Removing the common 5 mm observation variance in an independent
sensitivity calculation raises the mean only to `0.0027418519` nats; the largest
case remains at `0.0072111117` nats.

The local anisotropic component is nearly rank one, but the artifact contains
independent `(3, 3)` point blocks rather than a dense object covariance. A free
diagonal plus one three-vector factor requires six parameters, exactly as many as
a symmetric 3D covariance. This is local structure, not strict compression and
not evidence about whole-object joint tolerances.

Accordingly, `dependence_signal_supported`, `strict_compression_supported`, and
`headline_fused_claim_supported` are all false. The frozen protocol does not
authorize opening outcomes for the proposed empirical-covariance comparison.
