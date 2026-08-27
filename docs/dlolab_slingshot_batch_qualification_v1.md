# Native Slingshot Eight-Environment Qualification

This is a separate CPU execution optimization, not a method or reward change.
The native DLO-Lab environment already supports batched environments. Before
using that support for any optimizer, this frozen test compares an eight-world
batch with the completed isolated-process reference. It does not touch or alter
the concurrently running 24-action source bank.

Batch action indices are `[0,1,1,0,1,0,1,1]`, using the identical qualified zero
and pull controls. Native source, assets, solver settings, robot/controller,
seed, 900-step horizon, and reward remain identical. Each environment must agree
with its isolated reference within 1 micrometre for position traces and within
`rtol=1e-6, atol=1e-9` on every one of the 23 final native memory fields. Fixed
endpoints must stay within 1e-9 m, and native float32 reward must match exactly.
Boolean state and singleton environment axes are preserved, not cast or pooled.

The batch generation is sealed before arithmetic comparison. Any failure is
retained, without retries or relaxed tolerances. A pass qualifies only this
eight-environment execution shape and these controls; it does not demonstrate
task competence, optimizer superiority, or a Bayesian gain. A future optimizer
study must be separately source-frozen and retain all invalid control attempts.
There is no GPU work, new recording, protected data, target access, push, or merge.
