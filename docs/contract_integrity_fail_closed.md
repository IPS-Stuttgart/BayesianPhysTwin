# Fail-closed contract integrity

BayesianPhysTwin public artifacts use recursively frozen JSON metadata and
content-addressed identities. Two implementation details are security and
scientific-validity boundaries rather than convenience features:

1. validated metadata must not change after construction; and
2. physical closure evidence must contain at least one evaluated query.

## Sealed JSON containers

`frozen_finite_json_mapping` returns `FrozenDict` and `FrozenList` containers so
existing JSON, mapping, and sequence integrations remain compatible while
ordinary mutation syntax is rejected.

Python's built-in container types nevertheless expose base-class operations
that can bypass subclass mutation overrides. For example, these calls target
the backing storage directly:

```python
dict.__setitem__(metadata, "changed", True)
list.append(metadata["items"], "changed")
```

The frozen containers therefore retain a sealed snapshot of their validated
contents. Before an ordinary read or serialization operation, they verify:

- the exact item count and insertion order;
- the original key/value or element object identities; and
- the seals of nested `FrozenDict` and `FrozenList` values.

Detected backing-storage mutation raises `RuntimeError` rather than returning
either the pre-tamper or post-tamper interpretation.

The guarded read surface includes:

- indexing, iteration, membership, length, reversal, views, `get`, copying,
  representation, equality, and inequality;
- dictionary union in both operand orders;
- list `count`, `index`, concatenation, repetition, and ordering comparisons;
- `plain_json`, direct and nested `json.dumps`, and downstream
  content-addressed identity construction.

The guarded overrides conform to the repository's typed container contracts
without local MyPy suppressions.

Unsupported operations retain normal built-in behavior. For example, union
with a non-dictionary or concatenation with a non-list still raises
`TypeError`. Untampered containers remain compatible with ordinary `dict`,
`list`, JSON, equality, slicing, and detached-copy behavior.

Explicit calls to built-in base readers such as `dict.__getitem__` deliberately
operate below the public wrapper and are not an approved contract-consumption
route. Their existence is why every supported ordinary read path verifies the
seal before exposing data.

## Nonempty nonlinear closure evidence

`evaluate_nonlinear_closure` compares baseline, linearized, and nonlinear query
arrays. Shape `(0, 3)` previously satisfied the geometric shape check and
produced zero aggregate error, even though no physical query was evaluated.
That could incorrectly yield `candidate_valid=True`.

Closure evidence now requires shape `(Q, 3)` with `Q >= 1`. Empty replay is
rejected before any metric or validity result is constructed. A nonempty exact
replay remains valid and retains zero absolute and relative error.

## Compatibility

The hardening preserves existing untampered artifact bytes and ordinary
container compatibility. It changes no estimator, calibration threshold,
physical result, cohort, or recorded scientific outcome. Previously constructed
valid metadata remains readable; only backing-storage tampering and empty
closure evidence now fail closed.

These checks establish software artifact integrity. They do not establish
observation accuracy, covariance calibration, physical-query benefit,
deployment safety, or state of the art.
