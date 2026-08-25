# Recursive-corruption v2 CI registration

The focused `tests/test_recursive_corruption_benchmark_v2.py` suite is registered in the repository-wide `stable-core-coverage` manifest. The protected coverage ratchet therefore measures the new source module rather than treating it as an unexecuted changed file. This registration changes no estimator, threshold, seed roster, endpoint, or scientific result.
