# Task-objective uncertainty extension

The v2 controlled result adds uncertainty in the downstream objective to the
existing state-ambiguity atlas. For action-region half-spaces
`normal @ theta <= offset`, a translated task set `center + U` is certified
exactly when `normal @ center + support_U(normal) <= offset` for every
constraint.

The retained box experiment uses half-width `(0.1, 0.2)`. Over the valid center
domain, nominal capability covers 83.18% and all-objective capability covers
66.33%; the remaining 33.67% routes to fallback. The nominal task
`(-0.6, 0.1)` certifies `pull_left`, while its registered box with half-width
`(0.04, 0.05)` certifies no action.

This is deterministic mechanism evidence. It does not validate the objective,
task-uncertainty set, physical model, provider, safety, or deployment context.
