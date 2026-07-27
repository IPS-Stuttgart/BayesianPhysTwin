replace_once(
    "src/bayesian_phystwin/observation_belief.py",
    '''from collections.abc import Mapping
from collections.abc import Mapping as MappingABC
''',
    '''from collections.abc import Mapping
''',
)
replace_once(
    "src/bayesian_phystwin/observation_belief.py",
    "isinstance(value, MappingABC)",
    "isinstance(value, Mapping)",
)
replace_once(
    "CHANGELOG.md",
    '''- Missing private-Prob4D credentials now fail trusted pull requests, `main`,
    scheduled, and manual three-repository runs instead of producing a green skip.
    External-fork pull requests still run the producer-neutral consumer fixture and
    explicitly report that the secret-backed producer gate was unavailable and no
    current-Prob4D evidence was admitted.
''',
    '''- Missing private-Prob4D credentials now fail trusted pull requests, `main`,
  scheduled, and manual three-repository runs instead of producing a green skip.
  External-fork pull requests still run the producer-neutral consumer fixture and
  explicitly report that the secret-backed producer gate was unavailable and no
  current-Prob4D evidence was admitted.
''',
)
