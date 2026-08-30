# Query-Conditional Simulator Competence Atlas v3

V3 extends the immutable three-query v2 atlas with the terminal public DLO-Lab
separation development result. It does not modify any prior wrapping,
slingshot, or coiling metric or decision.

| Exact query | Native | Headroom | Source transfer | Prospective risk | Decision |
| --- | --- | --- | --- | --- | --- |
| Wrapping v9 | pass | pass | pass | pass | **certified** |
| Slingshot v2 | pass | fail | fail | fail | **rejected** |
| Coiling off-grid v2 | pass | fail | fail | not evaluated | **rejected** |
| Separation development v2 | fail | not evaluated | not evaluated | not evaluated | **rejected** |

The separation batch completed ordinary native execution but violated its
frozen material-attachment qualification: `30.452 mm` versus a `20 mm` cap.
The result stopped after one of nine worlds, before action-headroom analysis.
It therefore rejects only that exact observation/action/qualification query;
it is not a backend-wide conclusion about DLO-Lab separation.

This additional row makes the staged failure taxonomy concrete: a query can
fail before value is inspected, while coiling fails for insufficient action
headroom and Slingshot fails after reaching prospective risk evaluation.
Unknown or rejected queries retain the caller's exact baseline object.

The evidence uses only a public simulator, no new recordings or protected
targets. Arithmetic verification is not independent human review. Rebuild with:

```bash
PYTHONPATH=src python scripts/build_dlolab_query_competence_atlas_v3.py \
  --output /tmp/dlolab-query-competence-atlas-v3.json
```
