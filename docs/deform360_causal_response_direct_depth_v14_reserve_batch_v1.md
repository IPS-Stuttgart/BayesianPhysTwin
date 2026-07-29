# V14 reserve batch

The original V14 source campaign produced eight admissions from queue ranks
3--14. Four candidates were rejected by the frozen frame-zero admission
preflight. Those rejections are not model predictions, and the frozen queue
allows replacement only before the twelve-case source lock.

Before generating masks for any additional candidate, this child lock fixes
queue ranks 15--22 as one eight-candidate reserve batch. The final source panel
may take only the first four admissions in immutable queue order. Processing
later candidates in this batch does not permit skipping an earlier admission.

If the batch yields fewer than four admissions, another consecutive batch must
be fixed in a new child lock before its prefix masks or geometry are inspected.
No source outcome, future identity, target artifact, or held-v8 artifact is
part of this decision.
