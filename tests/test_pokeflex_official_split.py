import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.pokeflex_official_split import (
    OFFICIAL_LEGACY_TAKE_IDS,
    OFFICIAL_PUBLIC_CANDIDATE_TAKE_IDS,
    PUBLIC_ALIGNMENT_EXAMPLE_TAKE_IDS,
    UPSTREAM_PUBLIC_ALIGNMENT_COMMIT,
    audit_official_split,
    canonical_audit_sha256,
    discover_public_take_ids,
    public_take_id,
    write_official_split_audit,
)

PUBLIC_MAX_TAKE = {
    "3dPrintedBunny": 7,
    "3dPrintedCylinder": 6,
    "3dPrintedHeart": 6,
    "3dPrintedPizza": 6,
    "3dPrintedPyramid": 6,
    "Beanbag": 7,
    "FoamCylinder": 7,
    "FoamDice": 8,
    "FoamHalfSphere": 6,
    "MemoryFoam": 6,
    "Pillow": 7,
    "PlushDice": 8,
    "PlushMoon": 6,
    "PlushOctopus": 7,
    "PlushTurtle": 6,
    "PlushVolleyball": 6,
    "Sponge": 5,
    "ToiletPaperRoll": 6,
}

ROOT = Path(__file__).resolve().parents[1]
FROZEN_AUDIT = (
    ROOT
    / "results"
    / "sota"
    / "pokeflex_official_split_provenance_v2"
    / "public_inventory_audit.json"
)
FROZEN_AUDIT_SHA256 = (
    "c62f496ffea3e0dc9551cd7c9eb993ddf50eedea5f407d5a4fb51c7b153095c9"
)
FROZEN_AUDIT_FILE_SHA256 = (
    "fbc829724a8b6c53c44d28e6ec61dad3302fa39ed71ac369c5c900037b01b406"
)


def _public_inventory() -> tuple[str, ...]:
    return tuple(
        f"{object_name}_T{take_number}"
        for object_name, maximum in PUBLIC_MAX_TAKE.items()
        for take_number in range(1, maximum + 1)
    )


def test_object_projection_preserves_official_take_number() -> None:
    assert public_take_id("Fjadrar_T8") == "Pillow_T8"
    assert public_take_id("Heart_T14") == "3dPrintedHeart_T14"
    assert len(OFFICIAL_LEGACY_TAKE_IDS) == 18
    assert len(OFFICIAL_PUBLIC_CANDIDATE_TAKE_IDS) == 18


def test_public_inventory_requires_author_mapping_for_five_cases() -> None:
    result = audit_official_split(_public_inventory())
    split = result["official_split"]

    assert split["exact_public_match_count"] == 13
    assert split["unresolved_count"] == 5
    assert split["unresolved_public_identity_projections"] == [
        "Pillow_T8",
        "3dPrintedCylinder_T7",
        "3dPrintedHeart_T14",
        "Sponge_T10",
        "3dPrintedPizza_T13",
    ]
    assert split["exact_public_split_materializable"] is False
    assert result["decision"] == "author_mapping_or_processed_validation_set_required"
    assert result["audit_sha256"] == canonical_audit_sha256(result)


def test_lower_numbered_public_takes_do_not_resolve_internal_recordings() -> None:
    inventory = set(_public_inventory())
    inventory.update(
        {
            "Pillow_T7",
            "3dPrintedCylinder_T6",
            "3dPrintedHeart_T6",
            "Sponge_T5",
            "3dPrintedPizza_T6",
        }
    )

    result = audit_official_split(sorted(inventory))

    assert result["official_split"]["unresolved_count"] == 5
    assert result["official_split"]["exact_public_split_materializable"] is False


def test_exact_author_supplied_filenames_would_unlock_full_split() -> None:
    result = audit_official_split(OFFICIAL_PUBLIC_CANDIDATE_TAKE_IDS)

    assert result["official_split"]["exact_public_match_count"] == 18
    assert result["official_split"]["unresolved_count"] == 0
    assert result["official_split"]["exact_public_split_materializable"] is True
    assert result["decision"] == "full_official_split_available"


def test_upstream_public_alignment_is_explicitly_single_case() -> None:
    result = audit_official_split(_public_inventory())

    assert UPSTREAM_PUBLIC_ALIGNMENT_COMMIT == (
        "fa484b0fa94f59f51e8c5f2293a6b1bc378b7375"
    )
    assert PUBLIC_ALIGNMENT_EXAMPLE_TAKE_IDS == ("FoamDice_T3",)
    assert result["upstream"]["public_alignment_evaluation_take_ids"] == [
        "FoamDice_T3"
    ]
    assert "does not publish" in result["upstream"]["public_alignment_scope"]


def test_frozen_public_inventory_audit_is_canonical() -> None:
    audit = json.loads(FROZEN_AUDIT.read_text(encoding="utf-8"))

    assert hashlib.sha256(FROZEN_AUDIT.read_bytes()).hexdigest() == (
        FROZEN_AUDIT_FILE_SHA256
    )
    assert audit["audit_sha256"] == FROZEN_AUDIT_SHA256
    assert audit["audit_sha256"] == canonical_audit_sha256(audit)
    assert audit["public_inventory"]["take_count"] == 116
    assert audit["official_split"]["exact_public_match_count"] == 13
    assert audit["official_split"]["unresolved_count"] == 5
    assert audit["decision"] == "author_mapping_or_processed_validation_set_required"


def test_discovery_distinguishes_archives_from_empty_inventory_files(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "poking" / "Pillow" / "Pillow_T7.zip"
    real = tmp_path / "poking" / "FoamDice" / "FoamDice_T3.zip"
    empty.parent.mkdir(parents=True)
    real.parent.mkdir(parents=True)
    empty.touch()
    real.write_bytes(b"zip placeholder")

    assert discover_public_take_ids(tmp_path) == ("FoamDice_T3",)
    assert discover_public_take_ids(
        tmp_path,
        require_nonempty_archives=False,
    ) == ("FoamDice_T3", "Pillow_T7")


def test_written_audit_round_trips_with_canonical_digest(tmp_path: Path) -> None:
    for take_id in _public_inventory():
        archive = tmp_path / "poking" / take_id.rsplit("_T", 1)[0] / f"{take_id}.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b"source inventory")
    output = tmp_path / "audit.json"

    written = write_official_split_audit(tmp_path, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == written
    assert loaded["public_inventory"]["take_count"] == 116
    assert loaded["audit_sha256"] == canonical_audit_sha256(loaded)


@pytest.mark.parametrize("take_id", ["Pillow", "Pillow_T0", "Pillow_Tx"])
def test_invalid_take_ids_fail_closed(take_id: str) -> None:
    with pytest.raises(ValueError, match="invalid PokeFlex take ID"):
        audit_official_split([take_id])


def test_duplicate_inventory_entries_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate take IDs"):
        audit_official_split(["FoamDice_T3", "FoamDice_T3"])
