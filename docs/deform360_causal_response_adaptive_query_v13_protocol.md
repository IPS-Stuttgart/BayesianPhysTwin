# Causal-Response Adaptive Query V13

## Purpose

V12 failed before state inference: its exact twelve-camera carrier admitted two
of eight already-open source cases, while four other cases lacked one or more
complete registered streams. V13 tests one new carrier model without changing
V12 or reading any outcome:

1. certify every complete frame-zero camera from the registered panel;
2. select exactly eight complete cameras using only physical action support and
   frame-zero association support;
3. split them into deterministic, disjoint four-camera proposal and validation
   panels;
4. require three supported views in each panel when that fills the query budget;
5. otherwise permit two views in each panel only with fourfold covariance
   inflation and a fixed 5 mm shared-bias nuisance;
6. abstain exactly when even the inflated two-plus-two carrier is incomplete.

The panel objective is lexicographic: strict eligible identities, fallback
eligible identities, supported camera incidences, then association probability
mass. Exact ties select the lexicographically first panel. No tracker result,
innovation, target identity, or future metric participates.

## Development Boundary

The eight V10--V12 source cases have already been examined and can provide only
post-open development evidence. This feasibility run asks whether V13 can
materialize a carrier in at least six cases, including at least two strict-arm
admissions, with no technical failures. It does not test state-update accuracy,
calibration, transfer, confirmation, or state of the art.

Passing the carrier gate authorizes only a separately frozen source study of
the tactile event, tracker competence, bias-aware state update, and exact
baseline fallback. It does not authorize fresh-object selection. Fresh
Deform360 evaluation remains blocked until an independent held-v8 all-attempt
hash-only exclusion manifest is available.

## Causal and Calibration Boundary

- Object observations used for selection: frame zero only.
- Physical information: frame-zero state, graph basis, and source-sealed action
  support.
- Camera information: calibration, depth, and object mask at frame zero.
- Two-view support is never labeled equivalent to three-view support; its local
  covariance is multiplied by four.
- A 5 mm shared-bias standard deviation is declared separately from local
  triangulation uncertainty.
- No state update is constructed by this protocol.
- No tactile, tracker, future identity, future object observation, or future
  metric is read.
- V1 sealed targets and all held-v8 artifacts and processes remain untouched.
