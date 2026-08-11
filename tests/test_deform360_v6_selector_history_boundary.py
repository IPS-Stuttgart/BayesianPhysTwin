def test_selector_history_recovery_is_anchored_to_discovery_revision() -> None:
    with open(
        "scripts/ci/run_deform360_v6_source_prediction_evidence.sh",
        encoding="utf-8",
    ) as handle:
        text = handle.read()
    executable = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )

    assert 'DISCOVERY_REVISION="${CAUSAL4D_DISCOVERY_REVISION:-' in executable
    assert 'cat-file -e "${DISCOVERY_REVISION}^{commit}"' in executable
    assert "'+refs/heads/main:refs/remotes/origin/main'" in executable
    assert "'+refs/heads/*:refs/remotes/origin/*'" not in executable
    assert "--all --format='%H'" not in executable
    assert (
        '--format=\'%H\' "${DISCOVERY_REVISION}" -- "${SELECTOR_RELATIVE_PATH}"'
        in executable
    )
    assert '"${commit}:${SELECTOR_RELATIVE_PATH}"' in executable
    assert 'git -C "${repository}" archive' in executable
