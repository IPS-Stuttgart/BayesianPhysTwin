# JAX-FEM source-value gate v1

The source-physics-qualified JAX-FEM runtime is evaluated as a fixed,
equal-weight Poisson-ratio ensemble at `0.20`, `0.35`, and `0.45`, with Young's
modulus fixed at 100 kPa. Under this quasistatic displacement-only Dirichlet
formulation, absolute Young's modulus is structurally unidentifiable; the
source-physics gate verified that 25/100/500 kPa produces the same displacement
solution to numerical precision. Treating those values as an uncertainty
ensemble would therefore manufacture zero information.

All three full controller-driven predictions are sealed before prefix outcomes
are opened. No parameter or ensemble weight is selected from an object-motion
score. The 25 mm TET4 mesh and 15 mm rigid connected contact-patch projection
are inherited byte-for-byte from the qualified physics protocol. Full-horizon
physical checks are frozen from target-free action probes: contact-projection
error at most 20 mm, node displacement at most 350 mm, and deformation
determinants within `[0.5, 2.0]`.

The ensemble mean is the point prediction. The three members define an
equal-event 3D marginal energy score. Validation is the final third of each
already-open prefix; the first two thirds are reported as fit diagnostics only.
The validation gate requires:

1. at least 5% equal-group improvement over persistence on balanced point error;
2. no source group to regress against persistence;
3. at least 5% equal-group improvement in marginal energy score;
4. mean identity and Chamfer ratios no worse than 1.05 versus incumbent
   PhysTwin; and
5. finite, nondegenerate ensemble spread; and
6. the complete prediction grid passes the frozen full-horizon physical checks
   before any prefix outcome is read.

MatPhys is reported as a comparator but does not determine admission. A failed
gate returns the byte-exact incumbent archive for each group and leaves source
future outcomes unopened. Passing only authorizes the separately guarded future
score. This is an already-open two-action source mechanism study, not a fresh,
target, calibration, Causal4D, DEFORM, or state-of-the-art claim. The experiment
neither modifies nor retests DEFORM.
