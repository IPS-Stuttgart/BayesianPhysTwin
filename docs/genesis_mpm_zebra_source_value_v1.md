# Genesis MPM source-value gate v1

**Retained decision:** failed; exact incumbent fallback selected and source
future outcomes left unopened. See
[`genesis_mpm_zebra_source_value_v1_result.md`](genesis_mpm_zebra_source_value_v1_result.md).

The source-physics-qualified Genesis runtime is evaluated as a fixed,
equal-weight ensemble at 25, 100, and 500 kPa. All three full controller-driven
predictions are sealed before prefix outcomes are opened. No material parameter
or ensemble weight is selected from an object-motion score.

The ensemble mean is the point prediction. The three members define an
equal-event 3D marginal energy score. Validation is the final third of each
already-open prefix; the first two thirds are reported as fit diagnostics only.
The validation gate requires:

1. at least 5% equal-group improvement over persistence on balanced point error;
2. no source group to regress against persistence;
3. at least 5% equal-group improvement in marginal energy score;
4. mean identity and Chamfer ratios no worse than 1.05 versus incumbent
   PhysTwin; and
5. finite, nondegenerate ensemble spread.

MatPhys is reported as a comparator but does not determine admission. A failed
gate returns the byte-exact incumbent archive for each group and leaves source
future outcomes unopened. The experiment neither modifies nor retests DEFORM.
