# Deform360 cardinality source geometry gate v1

The independent `002-rope-silk` source run stopped at automatic geometry QA.
Four preregistered source episodes completed strict 81-frame Splatfacto
reconstruction, and all four failed at least one default `strict_hull_audit`
gate. Since the parent source gate requires every registered QA check to pass,
the gate became irrecoverably false and episodes 7 and 9 were stopped early.

No dense object trajectory was generated, no Warp rollout or physical-grid fit
was run, and no source-tail dynamics metric was read. Calibration episodes
`[3,4,8]` and sealed target episode `1` remain untouched.

| Episode | Chamfer (m) | Center error (m) | Major ratio | Middle span (m) | Failed gates |
|---:|---:|---:|---:|---:|---|
| 0 | 0.01136 | 0.02477 | 0.849 | 0.12759 | middle/minor span |
| 2 | 0.02718 | 0.05081 | 0.747 | 0.07335 | center |
| 5 | 0.02533 | 0.10919 | 0.896 | 0.22251 | center, middle/minor span |
| 6 | 0.02164 | 0.03758 | 0.613 | 0.11549 | major span ratio |

The pattern localizes the failure to transfer of the reconstruction QA model,
not to the not-yet-run cardinality-normalized physics hypothesis. The audit was
created for a nearly straight rope: it treats the second global PCA span as
object thickness and compares centers and major spans to a visual hull that can
contain large phantom volume. Those assumptions are not invariant to a curved
filament or multiview mask disagreement.

The next development step is a topology-aware, cross-view reconstruction audit:
use local filament thickness rather than global curvature, measure whether the
reconstruction projects inside observed masks, and use one-sided hull support
rather than forcing agreement with phantom hull volume. This `002-rope-silk`
source geometry is development data for that repair; a new independent object
is required before calling the repaired source gate prospective.
