# Temporary workflow cleanup scope

The branch-local carrier audit, alignment inspection, development runs, reserved
readiness audit, and reserved confirmation were all one-shot launchers. Their
result artifacts are retained by run/artifact identity. The launchers and
consumed request files must not be merged into the permanent workflow inventory.
