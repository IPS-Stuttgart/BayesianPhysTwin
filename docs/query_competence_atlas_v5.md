# Query-Conditional Simulator Competence Atlas v5

V5 adds the complete reward-aligned Slingshot v4 certificate without modifying
the rejected Slingshot v2 query or any other prior evidence.

| Exact query | Native | Headroom | Source transfer | Prospective risk | Decision |
| --- | --- | --- | --- | --- | --- |
| Wrapping v9 | pass | pass | pass | pass | **certified** |
| Slingshot v2 | pass | fail | fail | fail | **rejected** |
| Slingshot reward-aligned v4 | pass | pass | pass | pass | **certified** |
| Coiling off-grid v2 | pass | fail | fail | not evaluated | **rejected** |
| Separation development v2 | fail | not evaluated | not evaluated | not evaluated | **rejected** |
| Unknotting development v1 | fail | not evaluated | not evaluated | not evaluated | **rejected** |

Slingshot v4 uses 128 fresh calibration worlds and 288 fresh evaluation worlds.
All 3,328 independently launched native action processes completed ordinarily.
The guard changed 36/288 actions and improved reward by `0.0034568`, with paired
95% interval `[0.0015144, 0.0057113]`. Its harm upper bound is `0.04070`, and it
beats the matched simultaneous-regret guard by `0.0043380`, with paired interval
`[0.0019350, 0.0069734]`.

The two Slingshot entries are not contradictory labels for one backend. V2 and
v4 bind different observation policies and statistical units. V4 prospectively
treats rare late native-contact bifurcation as process variability while gating
reward repeatability; v2's rejected complete-belief query remains immutable.
This is precisely why the atlas certifies exact queries rather than simulator
names.

Rejected and unknown queries retain the caller's exact baseline object. The
evidence uses only public simulation, no new recordings or protected targets.
It is not an official benchmark, physical-safety, or backend-wide claim.
Rebuild with:

```bash
PYTHONPATH=src python scripts/build_dlolab_query_competence_atlas_v5.py \
  --output /tmp/dlolab-query-competence-atlas-v5.json
```
