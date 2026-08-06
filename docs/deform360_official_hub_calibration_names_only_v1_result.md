# Deform360 calibration names-only result v1

## Boundary

The first exact names-only plan was run at BayesianPhysTwin revision
`803787036f06c6c4a416cfd4e4220da05c32e14c`. It queried repository paths at the
locked Deform360 revision and opened no calibration RGB, tactile, robot,
confirmation, or outcome bytes.

The plan has canonical digest
`0494d23278be875ef18fd6be8eff0b68907c8ef853a2d170df6b51f3fadfe755` and file
SHA-256 `4db9572d75a336aaf732d364e6ba0c58ca46d0cd13a14aab8c2b529f85a8b992`.

## Initial result

The original planner admitted 3 of 10 calibration objects: one of five sheet
objects and two of five volumetric objects. All seven rejected objects exposed
the selected tactile recordings, but each of four tactile sensors contained
multiple timestamped `median_*.npy` baselines. The implementation interpreted
the protocol's one-baseline output contract as requiring one baseline in the
entire sensor directory and therefore rejected every such sensor.

## Target-free diagnosis

Filename-only analysis established the following across all ten calibration
objects and every selected tactile sensor:

- every recording and multi-candidate baseline had one parseable timestamp;
- the nearest baseline was unique in all cases;
- nearest absolute distance ranged from 15.92 to 419.66 seconds;
- nearest-versus-runner-up margin was at least 79.92 seconds; and
- the chosen baselines across sensors formed a capture cluster spanning 1.13 to
  2.70 seconds.

The amended planner therefore retains the sole baseline exactly or, for several
timestamped candidates, requires a unique nearest baseline within ten minutes,
a runner-up margin of at least one minute, and a cross-sensor capture span of at
most five seconds. The rule is deterministic, payload-free, and fails closed.
It does not replace objects or inspect target outcomes.

## Amended plan result

The amended implementation was frozen at
`a875aa0214dc48c054c116369662df3ec0d8f591` before its exact clean-checkout
names-only rerun. The amended plan passed with all 10 calibration objects,
including all five sheet and all five volumetric objects. Its canonical digest
is `3a5d2390546ea55370f45a688036e4911f0ffefc99cfed0732d498bc5e2cc5f4` and its
file SHA-256 is
`ef2d5d3be5f9e0373d7a305ad380b375e42c5003724c9eb05a8f5cec38b1c9d9`.

The rerun reproduced the same canonical and file digests as the development
rerun. It opened repository names only. No calibration payload was downloaded,
no confirmation object was inspected, and no target metric was computed. The
passing source-admission gate authorizes only the registered calibration
download and preparation stage.

## Claim boundary

This is source-admission and filename-association evidence only. It does not
establish tactile quality, provider competence, BayesianPhysTwin accuracy,
calibration, confirmation transfer, or state of the art.
