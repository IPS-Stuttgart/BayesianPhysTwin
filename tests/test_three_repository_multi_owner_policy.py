from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_three_repository_golden_path.sh"
DISCOVERY = ROOT / "scripts" / "discover_three_repository_tests.py"
WORKFLOW = ROOT / ".github" / "workflows" / "three-repository-test-discovery.yml"


def test_golden_path_collects_repository_owned_tests_from_all_three_sources() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert '"${BPT_BUILD_ROOT}/scripts/discover_three_repository_tests.py"' in text
    assert '--source "bayesian_phystwin=${BPT_BUILD_ROOT}"' in text
    assert '--source "prob4d=${PROB4D_BUILD_ROOT}"' in text
    assert '--source "causal4d=${CAUSAL4D_BUILD_ROOT}"' in text
    assert '--path-list "${TEST_PATH_LIST}"' in text
    assert '--inventory "${TEST_INVENTORY}"' in text
    assert 'mapfile -t integration_tests < "${TEST_PATH_LIST}"' in text
    assert (
        'export THREE_REPOSITORY_INTEGRATION_TEST_INVENTORY="${TEST_INVENTORY}"' in text
    )


def test_golden_path_uses_collision_safe_explicit_test_paths() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert 'for integration_test in "${integration_tests[@]}"; do' in text
    assert "--import-mode=prepend" in text
    assert '"${integration_test}"' in text
    assert "--import-mode=importlib" not in text
    assert (
        '"${BPT_BUILD_ROOT}"/integration_tests/test_three_repository_*.py' not in text
    )
    assert 'cp "${integration_tests[@]}" "${RUN_ROOT}/"' not in text
    assert "pytest -q test_three_repository_*.py" not in text


def test_discovery_helper_is_fail_closed_and_inventory_versioned() -> None:
    text = DISCOVERY.read_text(encoding="utf-8")

    assert (
        'INVENTORY_SCHEMA = "bayesian-phystwin.three-repository-test-inventory.v1"'
        in text
    )
    assert "flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL" in text
    assert "path.is_symlink()" in text
    assert "not path.is_file()" in text
    assert "if not pytest_paths:" in text
    assert "output path must not be inside source repository" in text
    assert "source owners must be unique" in text
    assert "source repository roots must be unique" in text


def test_focused_workflow_is_read_only_pinned_and_tracks_every_surface() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    required_paths = (
        ".github/workflows/three-repository-test-discovery.yml",
        "scripts/discover_three_repository_tests.py",
        "scripts/run_three_repository_golden_path.sh",
        "tests/test_three_repository_test_discovery.py",
        "tests/test_three_repository_multi_owner_policy.py",
        "docs/repository-owned-three-repository-tests.md",
    )
    for path in required_paths:
        assert text.count(f'- "{path}"') == 2

    assert "permissions:\n  contents: read" in text
    assert "persist-credentials: false" in text
    assert "actions/checkout@v" not in text
    assert "actions/setup-python@v" not in text
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in text
    assert "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97" in text
    assert "python -m ruff format --check" in text
    assert "tests/test_three_repository_test_discovery.py" in text
    assert "tests/test_three_repository_multi_owner_policy.py" in text
    assert "bash -n scripts/run_three_repository_golden_path.sh" in text
