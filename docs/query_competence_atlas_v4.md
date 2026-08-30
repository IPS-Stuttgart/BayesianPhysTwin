# Query-Conditional Simulator Competence Atlas v4

V4 extends the immutable four-query v3 atlas with the terminal public DLO-Lab
unknotting development result. It does not modify any prior metric or decision.

| Exact query | Native | Headroom | Source transfer | Prospective risk | Decision |
| --- | --- | --- | --- | --- | --- |
| Wrapping v9 | pass | pass | pass | pass | **certified** |
| Slingshot v2 | pass | fail | fail | fail | **rejected** |
| Coiling off-grid v2 | pass | fail | fail | not evaluated | **rejected** |
| Separation development v2 | fail | not evaluated | not evaluated | not evaluated | **rejected** |
| Unknotting development v1 | fail | not evaluated | not evaluated | not evaluated | **rejected** |

The unknotting batch completed ordinary native execution but violated its frozen
segment-length qualification: 18.11% maximum relative deviation versus a 10%
cap. An independent final-state reconstruction against the released rest geometry
still gives 15.75%. The result stopped after one of nine worlds, before action
headroom analysis. It rejects only that exact observation, action, and
qualification query; it is not a backend-wide conclusion about DLO-Lab
unknotting.

The attachment-offset correction introduced prospectively for this task passed
comfortably at 0.00022 mm drift. The rejection therefore does not reproduce the
separation task's absolute attachment-distance problem.

Unknown or rejected queries retain the caller's exact baseline object. The
evidence uses only a public simulator, no new recordings or protected targets.
Arithmetic verification is not independent human review. Rebuild with:

```bash
PYTHONPATH=src python scripts/build_dlolab_query_competence_atlas_v4.py \
  --output /tmp/dlolab-query-competence-atlas-v4.json
```
