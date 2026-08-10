#!/usr/bin/env python3
"""Apply final fail-closed hardening to the discrepancy tournament."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old_lines: list[str], new_lines: list[str]) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    old = "\n".join(old_lines)
    new = "\n".join(new_lines)
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement marker, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def patch_contracts() -> None:
    path = "src/bayesian_phystwin/_discrepancy_tournament_contracts.py"
    replace_once(
        path,
        [
            "DEFAULT_MAXIMUM_INPUT_BYTES: Final = 64 * 1024 * 1024",
            '_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?\\Z")',
        ],
        [
            "DEFAULT_MAXIMUM_INPUT_BYTES: Final = 64 * 1024 * 1024",
            "MAXIMUM_BOOTSTRAP_SAMPLES: Final = 100_000",
            "MAXIMUM_BOOTSTRAP_DRAW_ELEMENTS: Final = 50_000_000",
            "MAXIMUM_NUMERICAL_TOLERANCE: Final = 1e-9",
            "DISCREPANCY_TOURNAMENT_STATISTICAL_UNITS: Final[frozenset[str]] = (",
            "    frozenset(",
            "        {",
            '            "physical-object",',
            '            "independent-acquisition-session",',
            '            "physical-object-or-independent-acquisition-session",',
            "        }",
            "    )",
            ")",
            '_IDENTIFIER = re.compile(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?\\Z")',
        ],
    )
    replace_once(
        path,
        [
            "def _integer(value: object, *, name: str, minimum: int = 0) -> int:",
            "    if isinstance(value, bool) or not isinstance(value, int):",
            '        raise ValueError(f"{name} must be an integer")',
            "    if value < minimum:",
            '        raise ValueError(f"{name} must be at least {minimum}")',
            "    return value",
        ],
        [
            "def _integer(",
            "    value: object,",
            "    *,",
            "    name: str,",
            "    minimum: int = 0,",
            "    maximum: int | None = None,",
            ") -> int:",
            "    if isinstance(value, bool) or not isinstance(value, int):",
            '        raise ValueError(f"{name} must be an integer")',
            "    if value < minimum:",
            '        raise ValueError(f"{name} must be at least {minimum}")',
            "    if maximum is not None and value > maximum:",
            '        raise ValueError(f"{name} must be at most {maximum}")',
            "    return value",
        ],
    )
    replace_once(
        path,
        [
            "        bootstrap_samples=_integer(",
            '            payload["bootstrap_samples"],',
            '            name="selection.bootstrap_samples",',
            "            minimum=100,",
            "        ),",
        ],
        [
            "        bootstrap_samples=_integer(",
            '            payload["bootstrap_samples"],',
            '            name="selection.bootstrap_samples",',
            "            minimum=100,",
            "            maximum=MAXIMUM_BOOTSTRAP_SAMPLES,",
            "        ),",
        ],
    )
    replace_once(
        path,
        [
            "        numerical_tolerance=_number(",
            '            payload["numerical_tolerance"],',
            '            name="selection.numerical_tolerance",',
            "            minimum=0.0,",
            "        ),",
        ],
        [
            "        numerical_tolerance=_number(",
            '            payload["numerical_tolerance"],',
            '            name="selection.numerical_tolerance",',
            "            minimum=0.0,",
            "            maximum=MAXIMUM_NUMERICAL_TOLERANCE,",
            "        ),",
        ],
    )
    replace_once(
        path,
        [
            "        _require(",
            "            fallback_record.proper_score == fallback_record.fallback_proper_score,",
            '            f"{unit_id} fallback raw proper score changed",',
            "        )",
            "",
            '    evaluation = _parse_evaluation(payload["evaluation"])',
        ],
        [
            "        _require(",
            "            fallback_record.proper_score == fallback_record.fallback_proper_score,",
            '            f"{unit_id} fallback raw proper score changed",',
            "        )",
            "        for record in unit_records:",
            "            if record.accepted:",
            "                continue",
            "            _require(",
            "                record.interval_covered == fallback_record.interval_covered,",
            '                f"{unit_id} rejected interval coverage differs from exact fallback",',
            "            )",
            "            _require(",
            "                record.interval_width == fallback_record.interval_width,",
            '                f"{unit_id} rejected interval width differs from exact fallback",',
            "            )",
            "",
            '    evaluation = _parse_evaluation(payload["evaluation"])',
        ],
    )
    replace_once(
        path,
        [
            "    _require(",
            "        len(groups) >= required_groups,",
            '        "tournament has too few independent groups for the frozen selection rule",',
            "    )",
            "",
            "    interval_presence: dict[str, set[bool]] = {",
        ],
        [
            "    _require(",
            "        len(groups) >= required_groups,",
            '        "tournament has too few independent groups for the frozen selection rule",',
            "    )",
            "    bootstrap_draw_elements = (",
            "        2",
            "        * len(candidates)",
            "        * config.bootstrap_samples",
            "        * len(groups)",
            "        * len(groups)",
            "    )",
            "    _require(",
            "        bootstrap_draw_elements <= MAXIMUM_BOOTSTRAP_DRAW_ELEMENTS,",
            '        "bootstrap work budget exceeded; reduce samples, candidates, or groups",',
            "    )",
            "",
            "    interval_presence: dict[str, set[bool]] = {",
        ],
    )
    replace_once(
        path,
        [
            "        _require(not missing, f\"interval coverage is missing for candidates {missing}\")",
            "",
            "    return TournamentEvidence(",
        ],
        [
            "        _require(not missing, f\"interval coverage is missing for candidates {missing}\")",
            "",
            "    statistical_unit = _text(",
            '        payload["statistical_unit"],',
            '        name="statistical_unit",',
            "    )",
            "    _require(",
            "        statistical_unit in DISCREPANCY_TOURNAMENT_STATISTICAL_UNITS,",
            '        "statistical_unit must identify physical objects or independent sessions",',
            "    )",
            "",
            "    return TournamentEvidence(",
        ],
    )
    replace_once(
        path,
        [
            '        statistical_unit=_text(payload["statistical_unit"], name="statistical_unit"),'
        ],
        ["        statistical_unit=statistical_unit,"],
    )


def patch_analysis() -> None:
    path = "src/bayesian_phystwin/_discrepancy_tournament_analysis.py"
    replace_once(
        path,
        [
            "from .provider_failure_report_io import canonical_json_sha256",
            "",
            "",
            "def _records_for(",
        ],
        [
            "from .provider_failure_report_io import canonical_json_sha256",
            "",
            "_MAXIMUM_BOOTSTRAP_CHUNK_ELEMENTS = 100_000",
            "",
            "",
            "def _records_for(",
        ],
    )
    replace_once(
        path,
        [
            "    differences = candidate - reference",
            "    rng = np.random.default_rng(seed)",
            "    indices = rng.integers(0, len(differences), size=(samples, len(differences)))",
            "    estimates = np.mean(differences[indices], axis=1)",
            "    return [",
        ],
        [
            "    differences = candidate - reference",
            "    rng = np.random.default_rng(seed)",
            "    estimates = np.empty(samples, dtype=np.float64)",
            "    chunk_size = max(",
            "        1,",
            "        min(",
            "            samples,",
            "            _MAXIMUM_BOOTSTRAP_CHUNK_ELEMENTS // len(differences),",
            "        ),",
            "    )",
            "    for start in range(0, samples, chunk_size):",
            "        stop = min(samples, start + chunk_size)",
            "        indices = rng.integers(",
            "            0,",
            "            len(differences),",
            "            size=(stop - start, len(differences)),",
            "        )",
            "        estimates[start:stop] = np.mean(differences[indices], axis=1)",
            "    return [",
        ],
    )


def patch_tests() -> None:
    path = "tests/test_horizon_conditioned_discrepancy_tournament.py"
    replace_once(
        path,
        ['        "statistical_unit": "physical-object-or-session",'],
        [
            '        "statistical_unit": (',
            '            "physical-object-or-independent-acquisition-session"',
            "        ),",
        ],
    )
    replace_once(
        path,
        ["def test_information_boundary_and_interval_coverage_fail_closed() -> None:"],
        [
            "@pytest.mark.parametrize(",
            '    ("field", "value", "message"),',
            "    [",
            "        (",
            '            "interval_covered",',
            "            False,",
            '            "rejected interval coverage differs from exact fallback",',
            "        ),",
            "        (",
            '            "interval_width",',
            "            1.0,",
            '            "rejected interval width differs from exact fallback",',
            "        ),",
            "    ],",
            ")",
            "def test_rejected_candidate_must_retain_exact_fallback_interval(",
            "    field: str,",
            "    value: object,",
            "    message: str,",
            ") -> None:",
            "    payload = _payload()",
            '    records = payload["records"]',
            "    assert isinstance(records, list)",
            "    row = next(",
            "        record",
            "        for record in records",
            '        if record["candidate_id"] == "structured"',
            '        and record["group_id"] == "group-0"',
            "    )",
            '    row["accepted"] = False',
            '    row["deployed_point_loss"] = row["fallback_point_loss"]',
            '    row["deployed_proper_score"] = row["fallback_proper_score"]',
            "    row[field] = value",
            "",
            "    with pytest.raises(ValueError, match=message):",
            "        parse_discrepancy_candidate_tournament(payload)",
            "",
            "",
            "def test_information_boundary_and_interval_coverage_fail_closed() -> None:",
        ],
    )
    replace_once(
        path,
        ["def test_interval_free_non_crossfit_tournament_is_supported() -> None:"],
        [
            "def test_statistical_unit_and_numerical_limits_fail_closed() -> None:",
            "    payload = _payload()",
            '    payload["statistical_unit"] = "frame"',
            "    with pytest.raises(ValueError, match=\"statistical_unit must identify\"):",
            "        parse_discrepancy_candidate_tournament(payload)",
            "",
            "    payload = _payload()",
            '    selection = payload["selection"]',
            "    assert isinstance(selection, dict)",
            '    selection["numerical_tolerance"] = 1e-6',
            "    with pytest.raises(ValueError, match=\"at most 1e-09\"):",
            "        parse_discrepancy_candidate_tournament(payload)",
            "",
            "    payload = _payload()",
            '    selection = payload["selection"]',
            "    assert isinstance(selection, dict)",
            '    selection["bootstrap_samples"] = 100_001',
            "    with pytest.raises(ValueError, match=\"at most 100000\"):",
            "        parse_discrepancy_candidate_tournament(payload)",
            "",
            "",
            "def test_bootstrap_work_budget_fails_closed() -> None:",
            "    payload = _payload()",
            '    records = payload["records"]',
            '    selection = payload["selection"]',
            "    assert isinstance(records, list)",
            "    assert isinstance(selection, dict)",
            "    templates = [",
            "        deepcopy(record)",
            "        for record in records",
            '        if record["group_id"] == "group-0"',
            "    ]",
            "    for index in range(6, 10):",
            '        group = f"group-{index}"',
            "        for template in templates:",
            "            record = deepcopy(template)",
            '            record["group_id"] = group',
            '            record["unit_id"] = f"{group}-endpoint"',
            "            records.append(record)",
            '    selection["bootstrap_samples"] = 100_000',
            "",
            "    with pytest.raises(ValueError, match=\"bootstrap work budget exceeded\"):",
            "        parse_discrepancy_candidate_tournament(payload)",
            "",
            "",
            "def test_interval_free_non_crossfit_tournament_is_supported() -> None:",
        ],
    )


def patch_docs() -> None:
    path = "docs/discrepancy_candidate_tournament_v1.md"
    replace_once(
        path,
        [
            "The statistical group must be a physical object or independent acquisition",
            "session, not a frame, point, track, camera, or taxel. All point losses, proper",
            "scores, coverage values, and widths are averaged within a group before groups",
            "receive equal weight.",
        ],
        [
            "The statistical-unit field uses a closed vocabulary: physical object,",
            "independent acquisition session, or an explicitly mixed object/session roster.",
            "A frame, point, track, camera, or taxel declaration is rejected. All point",
            "losses, proper scores, coverage values, and widths are averaged within a group",
            "before groups receive equal weight.",
            "",
            "Bootstrap samples, total candidate/fold resampling draws, and the numerical",
            "comparison tolerance are resource-bounded by the contract. Bootstrap draws are",
            "materialized in deterministic chunks, so a valid input cannot request one dense",
            "samples-by-groups allocation.",
        ],
    )
    replace_once(
        path,
        [
            "A rejected candidate must deploy the exact physical-fallback values. The",
            "registered physical-fallback candidate must itself be rejected and reproduce its",
            "raw fallback values exactly.",
        ],
        [
            "A rejected candidate must deploy the exact physical-fallback point loss, proper",
            "score, interval coverage decision, and complete interval width. The registered",
            "physical-fallback candidate must itself be rejected and reproduce its raw",
            "fallback values exactly.",
        ],
    )


def main() -> None:
    patch_contracts()
    patch_analysis()
    patch_tests()
    patch_docs()


if __name__ == "__main__":
    main()
