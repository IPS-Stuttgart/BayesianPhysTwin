from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

import bayesian_phystwin.cli.deform360_adaptive_covariance_confirmation_lock as lock_cli
import bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock as lock


H1 = "a" * 40
OTHER_H1 = "b" * 40


def _selected(
    payload: dict[str, object],
) -> dict[str, list[tuple[str, list[int]]]]:
    cohort = payload["cohort"]
    assert isinstance(cohort, dict)
    return {
        stratum: [
            (
                record["object_id"],
                [episode["episode_id"] for episode in record["episodes"]],
            )
            for record in cohort[stratum]
        ]
        for stratum in lock.EXPECTED_STRATA
    }


def test_exact_audited_metadata_bindings_and_name_only_eligibility() -> None:
    assert lock.DATASET_REVISION == "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
    assert lock.RAW_TREE_ID == "a4a36e0669bcc86ab79e7ffb35aada7f2334c570"
    assert lock.OBJECT_INVENTORY_SHA256 == (
        "cf82fec6c715c3dafa2f3cadbeb6402f68443305fc40ff418075fe2cc22febb2"
    )
    assert lock.TAXONOMY_SOURCE_COMMIT == ("d72ca1dee49841b7d3b020da4380d0ef0a3f7d7c")
    assert lock.TAXONOMY_SOURCE_SHA256 == (
        "9e933a93e28869cc67101300cd0990feb148841355f6a2416dc8cb92f595fa01"
    )
    assert {key: len(value) for key, value in lock.NAME_ONLY_TAXONOMY.items()} == {
        "filament": 12,
        "sheet": 77,
        "volumetric": 39,
    }
    assert len(lock.EXCLUDED_OBJECT_IDS) == 31
    exclusion_payload = "".join(
        object_id + "\n" for object_id in lock.EXCLUDED_OBJECT_IDS
    ).encode()
    assert hashlib.sha256(exclusion_payload).hexdigest() == (
        "1c937d23a37d9a330e157c0ad40a92131775c378a856364d5fa489a2b52c0bd1"
    )
    assert lock.eligible_objects("filament") == ("181-belt",)
    assert {
        stratum: len(lock.eligible_objects(stratum)) for stratum in lock.EXPECTED_STRATA
    } == {"filament": 1, "sheet": 70, "volumetric": 32}


def test_seed_and_framed_rank_have_independent_known_vectors() -> None:
    expected_seed = hashlib.sha256(
        lock.PROTOCOL_ID.encode()
        + b"\0"
        + H1.encode()
        + b"\0"
        + lock.DATASET_REVISION.encode()
    ).hexdigest()
    assert expected_seed == (
        "2170c1dfa77cbce0a438b7f7e80510dc4909c8a3b5c0a43a8ebe55a171620b73"
    )
    assert lock.selection_seed_sha256(H1) == expected_seed

    frames = (b"alpha", b"", b"omega")
    independently_framed = b"".join(
        len(frame).to_bytes(8, "big") + frame for frame in frames
    )
    assert (
        lock.framed_sha256(*frames) == hashlib.sha256(independently_framed).hexdigest()
    )


def test_h1_deterministically_freezes_exact_17_objects_and_34_cases() -> None:
    first = lock.build_confirmation_cohort_lock(H1)
    second = lock.build_confirmation_cohort_lock(H1)

    assert first == second
    assert first["artifact_sha256"] == (
        "b0d0395e388e4b5c849c9fb9abd2d23e960e3333300a19686d186d6a71f5ae53"
    )
    assert _selected(first) == {
        "filament": [("181-belt", [1, 3])],
        "sheet": [
            ("173-poster-paper-cloth", [4, 2]),
            ("122-sheets-cloth", [3, 7]),
            ("167-glove-gray-cloth", [8, 5]),
            ("176-candy-packet-cloth", [5, 4]),
            ("082-curtain-cloth", [4, 1]),
            ("118-envelope-cloth", [0, 6]),
            ("027-umbrella-bag-cloth", [2, 4]),
            ("166-glove-green-cloth", [6, 7]),
        ],
        "volumetric": [
            ("050-boxing", [9, 8]),
            ("188-foam-roll-small", [8, 3]),
            ("196-hello-kitty-white", [4, 5]),
            ("138-sponge-stamps", [5, 6]),
            ("152-slime", [1, 9]),
            ("102-stress-ball", [5, 1]),
            ("153-cake", [7, 6]),
            ("192-fish", [6, 8]),
        ],
    }
    assert first["case_count"] == 34
    assert len(set(first["selected_case_ids"])) == 34
    validation = lock.validate_confirmation_cohort_lock(
        first,
        expected_implementation_commit_h1=H1,
    )
    assert validation["object_count"] == 17
    assert validation["case_count"] == 34


def test_h1_is_a_real_selection_input_and_must_be_full_lowercase() -> None:
    first = lock.build_confirmation_cohort_lock(H1)
    other = lock.build_confirmation_cohort_lock(OTHER_H1)
    assert (
        first["two_commit_freeze"]["selection_seed_sha256"]
        != (other["two_commit_freeze"]["selection_seed_sha256"])
    )
    assert first["selected_case_ids"] != other["selected_case_ids"]

    for invalid in (
        "a" * 39,
        "A" * 40,
        "refs/heads/main",
        "0" * 40,
        "g" * 40,
    ):
        with pytest.raises(ValueError, match="H1 must be"):
            lock.build_confirmation_cohort_lock(invalid)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["fresh_identity_exclusions"].update(
                {"object_ids": payload["fresh_identity_exclusions"]["object_ids"][1:]}
            ),
            "checksum mismatch",
        ),
        (
            lambda payload: payload["camera_budget_semantics"].update(
                {"unit": "total hardware cameras"}
            ),
            "checksum mismatch",
        ),
        (
            lambda payload: payload["cohort"]["sheet"][0].update(
                {"object_id": "010-orange-cloth"}
            ),
            "checksum mismatch",
        ),
        (
            lambda payload: payload.update({"unexpected": True}),
            "checksum mismatch",
        ),
    ],
)
def test_any_unsigned_lock_mutation_fails_closed(
    mutation: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    payload = lock.build_confirmation_cohort_lock(H1)
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        lock.validate_confirmation_cohort_lock(payload)


def test_resigned_mutation_and_wrong_expected_h1_are_rejected() -> None:
    payload = lock.build_confirmation_cohort_lock(H1)
    payload["taxonomy_imbalance_disclosure"]["filament_statement"] = (
        "one episode is enough for a population claim"
    )
    payload["artifact_sha256"] = lock.cohort_lock_sha256(payload)
    with pytest.raises(ValueError, match="deterministic H1-derived lock"):
        lock.validate_confirmation_cohort_lock(payload)

    pristine = lock.build_confirmation_cohort_lock(H1)
    with pytest.raises(ValueError, match="lock H1 changed"):
        lock.validate_confirmation_cohort_lock(
            pristine,
            expected_implementation_commit_h1=OTHER_H1,
        )


def test_atomic_writer_creates_absent_lock_and_never_overwrites(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "locks" / "confirmation.json"
    payload = lock.write_confirmation_cohort_lock(destination, H1)

    assert json.loads(destination.read_text()) == payload
    assert (
        lock.load_confirmation_cohort_lock(
            destination,
            expected_implementation_commit_h1=H1,
        )
        == payload
    )
    original = destination.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        lock.write_confirmation_cohort_lock(destination, OTHER_H1)
    assert destination.read_bytes() == original
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_validator_rejects_malformed_freeze_object_as_value_error() -> None:
    payload = copy.deepcopy(lock.build_confirmation_cohort_lock(H1))
    payload["two_commit_freeze"] = []
    payload["artifact_sha256"] = lock.cohort_lock_sha256(payload)
    with pytest.raises(ValueError, match="two_commit_freeze"):
        lock.validate_confirmation_cohort_lock(payload)


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.json"
    source.write_text('{"schema_version":1,"schema_version":1}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        lock.load_confirmation_cohort_lock(source)


def test_lock_cli_validates_exact_h1_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[tuple[object, ...]] = []

    def validate(*args: object, **kwargs: object) -> dict[str, object]:
        events.append(("validate", args, kwargs))
        return {"implementation_commit_h1": H1}

    def write(path: str, h1: str) -> dict[str, object]:
        events.append(("write", path, h1))
        payload = lock.build_confirmation_cohort_lock(h1)
        return payload

    monkeypatch.setattr(
        lock_cli,
        "validate_confirmation_h1_lock_generation_entrypoint",
        validate,
    )
    monkeypatch.setattr(lock_cli, "write_confirmation_cohort_lock", write)
    lock_cli.main(
        [
            "--adapter-repo",
            "/adapter",
            "--implementation-commit-h1",
            H1,
            "--output",
            "/adapter/configs/sota/lock.json",
        ]
    )

    assert events == [
        (
            "validate",
            (
                "/adapter",
                "/adapter/configs/sota/lock.json",
                H1,
            ),
            {
                "entrypoint_file": lock_cli.__file__,
                "entrypoint_repository_path": lock_cli.ENTRYPOINT_REPOSITORY_PATH,
            },
        ),
        (
            "write",
            "/adapter/configs/sota/lock.json",
            H1,
        ),
    ]
    output = json.loads(capsys.readouterr().out)
    assert output["implementation_commit_h1"] == H1
