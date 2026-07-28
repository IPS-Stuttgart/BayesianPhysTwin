from __future__ import annotations

from pathlib import Path


def replace_exact(text: str, old: str, new: str, *, count: int, name: str) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"{name}: expected {count} anchors, found {actual}")
    return text.replace(old, new)


def main() -> None:
    path = Path('.github/workflows/tests.yml')
    text = path.read_text(encoding='utf-8')

    repeated = (
        '            tests/test_causal4d_graph_provider_v1.py \\\n'
        '            tests/test_causal4d_provider_v1.py \\\n'
        '            tests/test_gauge_aware_belief.py \\\n'
    )
    repeated_replacement = (
        '            tests/test_causal4d_graph_provider_v1.py \\\n'
        '            tests/test_causal4d_provider_v1.py \\\n'
        '            tests/test_causal4d_public_provider_v1.py \\\n'
        '            tests/test_legacy_artifacts.py \\\n'
        '            tests/test_gauge_aware_belief.py \\\n'
    )
    text = replace_exact(
        text,
        repeated,
        repeated_replacement,
        count=2,
        name='stable coverage and core contracts',
    )

    stable_files = (
        '            src/bayesian_phystwin/_gauge_aware_solver.py\n'
        '            src/bayesian_phystwin/causal4d_provider_v1.py\n'
        '            src/bayesian_phystwin/cli/main.py\n'
    )
    stable_files_replacement = (
        '            src/bayesian_phystwin/_gauge_aware_solver.py\n'
        '            src/bayesian_phystwin/causal4d_artifacts_v1.py\n'
        '            src/bayesian_phystwin/causal4d_provider_v1.py\n'
        '            src/bayesian_phystwin/causal4d_public_provider_v1.py\n'
        '            src/bayesian_phystwin/cli/main.py\n'
    )
    text = replace_exact(
        text,
        stable_files,
        stable_files_replacement,
        count=1,
        name='stable file inventory',
    )

    provider = (
        '            tests/test_causal4d_graph_provider_v1.py \\\n'
        '            tests/test_causal4d_graph_provider_parity.py \\\n'
        '            tests/test_causal4d_provider_v1.py \\\n'
        '            tests/test_gauge_aware_belief.py \\\n'
    )
    provider_replacement = (
        '            tests/test_causal4d_graph_provider_v1.py \\\n'
        '            tests/test_causal4d_graph_provider_parity.py \\\n'
        '            tests/test_causal4d_provider_v1.py \\\n'
        '            tests/test_causal4d_public_provider_v1.py \\\n'
        '            tests/test_legacy_artifacts.py \\\n'
        '            tests/test_gauge_aware_belief.py \\\n'
    )
    text = replace_exact(
        text,
        provider,
        provider_replacement,
        count=1,
        name='provider contract suite',
    )

    path.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
