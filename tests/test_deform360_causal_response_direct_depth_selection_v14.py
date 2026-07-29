from __future__ import annotations

from pathlib import Path

import pytest

from bayesian_phystwin.deform360_causal_response_direct_depth_selection_v14 import (
    V14SelectionDisposition,
    build_v14_selection_ledger,
    validate_v14_selection_ledger,
    write_v14_selection_ledger,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _dispositions(
    *,
    rejected_prefix: int = 2,
) -> tuple[V14SelectionDisposition, ...]:
    rows = []
    for index in range(rejected_prefix + 12):
        admitted = index >= rejected_prefix
        rows.append(
            V14SelectionDisposition(
                queue_rank=index + 1,
                object_hash=_digest(100 + index),
                case_hash=_digest(200 + index),
                status="admitted" if admitted else "technical_preflight_failure",
                disposition_artifact_sha256=_digest(300 + index),
                disposition_file_sha256=_digest(400 + index),
                selected=admitted,
            )
        )
    return tuple(rows)


def test_v14_selection_ledger_round_trip(tmp_path: Path) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text("{}")
    admission = tmp_path / "admission.json"
    admission.write_text("{}")
    ledger = build_v14_selection_ledger(
        _dispositions(),
        repository_revision="a" * 40,
        queue_sha256="b" * 64,
        queue_path=queue,
        admission_prelock_config_sha256="c" * 64,
        admission_prelock_path=admission,
    )
    output = tmp_path / "selection.json"

    write_v14_selection_ledger(output, ledger)

    assert validate_v14_selection_ledger(output) == ledger
    assert ledger.dispositions[-1].queue_rank == 14
    assert sum(item.selected for item in ledger.dispositions) == 12


def test_v14_selection_rejects_a_queue_gap(tmp_path: Path) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text("{}")
    admission = tmp_path / "admission.json"
    admission.write_text("{}")
    dispositions = list(_dispositions())
    del dispositions[5]

    with pytest.raises(ValueError, match="contiguous"):
        build_v14_selection_ledger(
            dispositions,
            repository_revision="a" * 40,
            queue_sha256="b" * 64,
            queue_path=queue,
            admission_prelock_config_sha256="c" * 64,
            admission_prelock_path=admission,
        )


def test_v14_selection_rejects_more_than_twelve_admissions(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text("{}")
    admission = tmp_path / "admission.json"
    admission.write_text("{}")
    dispositions = list(_dispositions())
    dispositions.append(
        V14SelectionDisposition(
            queue_rank=15,
            object_hash=_digest(500),
            case_hash=_digest(501),
            status="admitted",
            disposition_artifact_sha256=_digest(502),
            disposition_file_sha256=_digest(503),
            selected=True,
        )
    )

    with pytest.raises(ValueError, match="twelfth"):
        build_v14_selection_ledger(
            dispositions,
            repository_revision="a" * 40,
            queue_sha256="b" * 64,
            queue_path=queue,
            admission_prelock_config_sha256="c" * 64,
            admission_prelock_path=admission,
        )
