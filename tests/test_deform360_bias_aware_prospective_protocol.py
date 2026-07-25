import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesian_phystwin.deform360_bias_aware_prospective_download import (
    RELEASED_METADATA_OBJECT_ALIASES,
    bias_aware_prospective_download_plan,
    build_bias_aware_download_manifest,
    download_bias_aware_prospective_panel_by_object,
    validate_bias_aware_download_root,
)
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import (
    EXPECTED_CALIBRATION_COHORT,
    EXPECTED_STRATA,
    EXPECTED_TARGET_COHORT,
    OPEN_OR_RESERVED_OBJECTS,
    load_bias_aware_prospective_protocol,
    metadata_ranked_episode_ids,
    metadata_ranked_objects,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY_ROOT
    / "configs"
    / "sota"
    / "deform360_bias_aware_guarded_belief_prospective_v1.json"
)


def test_prospective_protocol_locks_fresh_metadata_ranked_cohorts() -> None:
    protocol = load_bias_aware_prospective_protocol(PROTOCOL)

    assert protocol["config_sha256"] == (
        "b6b19be5eaadf830a77f36cccddd38f5b7a35527ca21f7743d2ef147fceabbce"
    )
    calibration = protocol["calibration_cohort"]
    target = protocol["target_cohort"]
    assert calibration == EXPECTED_CALIBRATION_COHORT
    assert target == EXPECTED_TARGET_COHORT
    calibration_objects = {
        object_id for records in calibration.values() for object_id in records
    }
    target_objects = {
        object_id for records in target.values() for object_id in records
    }
    assert len(calibration_objects) == 9
    assert len(target_objects) == 12
    assert not calibration_objects & target_objects
    assert not (calibration_objects | target_objects) & OPEN_OR_RESERVED_OBJECTS
    assert len(OPEN_OR_RESERVED_OBJECTS) == 40

    for stratum in EXPECTED_STRATA:
        ranked = metadata_ranked_objects(stratum)
        assert tuple(calibration[stratum]) == ranked[:3]
        assert tuple(target[stratum]) == ranked[3:7]
        for object_id, episodes in calibration[stratum].items():
            assert episodes == metadata_ranked_episode_ids(
                object_id, "calibration", 1
            )
        for object_id, episodes in target[stratum].items():
            assert episodes == metadata_ranked_episode_ids(object_id, "target", 2)


def test_prospective_method_binding_matches_committed_source_artifacts() -> None:
    protocol = load_bias_aware_prospective_protocol(PROTOCOL)
    method = protocol["config"]["method"]
    bound_paths = {
        "bias_aware_belief.py": (
            REPOSITORY_ROOT / "src" / "bayesian_phystwin" / "bias_aware_belief.py"
        ),
        "deform360_bias_aware_belief_development.py": (
            REPOSITORY_ROOT
            / "src"
            / "bayesian_phystwin"
            / "deform360_bias_aware_belief_development.py"
        ),
        "deform360_raw_pairwise_correspondence_diagnostic.py": (
            REPOSITORY_ROOT
            / "src"
            / "bayesian_phystwin"
            / "deform360_raw_pairwise_correspondence_diagnostic.py"
        ),
        "deform360_raw_camera_observation.py": (
            REPOSITORY_ROOT
            / "src"
            / "bayesian_phystwin"
            / "deform360_raw_camera_observation.py"
        ),
    }
    for name, path in bound_paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == method[
            "implementation_sha256"
        ][name]

    result_root = (
        REPOSITORY_ROOT
        / "results"
        / "sota"
        / "deform360_bias_aware_guarded_belief_v4"
    )
    assert hashlib.sha256((result_root / "summary.json").read_bytes()).hexdigest() == (
        method["source_summary_sha256"]
    )
    assert hashlib.sha256(
        (result_root / "prospective_lock.json").read_bytes()
    ).hexdigest() == method["source_lock_sha256"]


def test_calibration_may_only_refit_bound_and_blocks_target_on_failure() -> None:
    protocol = load_bias_aware_prospective_protocol(PROTOCOL)
    gate = protocol["config"]["calibration_gate"]
    target = protocol["config"]["target_evaluation"]

    assert gate["permitted_change"] == (
        "refit only the direct source-group regret bound"
    )
    assert gate["minimum_new_eligible_object_groups"] == 5
    assert gate["minimum_combined_eligible_object_groups"] == 9
    assert gate["required_finite_sample_coverage"] == 0.90
    assert gate["target_access_if_gate_fails"] == "forbidden"
    assert not gate["quality_failure_replacement_allowed"]
    assert not target["replacement_allowed"]
    assert not target["direct_official_sota_claim_allowed"]


def test_download_plan_contains_only_locked_objects() -> None:
    plan = bias_aware_prospective_download_plan(PROTOCOL)

    assert len(plan.calibration_objects) == 9
    assert len(plan.target_objects) == 12
    assert len(plan.object_ids) == 21
    assert len(plan.allow_patterns) == 21
    assert set(plan.allow_patterns) == {
        f"raw/{object_id}/*" for object_id in plan.object_ids
    }
    assert plan.ignore_patterns == ("*.flac", "*.wav")


def test_download_root_rejects_unlocked_object(tmp_path: Path) -> None:
    plan = bias_aware_prospective_download_plan(PROTOCOL)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / plan.object_ids[0]).mkdir()
    (raw / "001-rope").mkdir()

    with pytest.raises(ValueError, match="unlocked objects"):
        validate_bias_aware_download_root(
            tmp_path, plan=plan, require_complete=False
        )


def test_download_manifest_preserves_roles_and_information_boundary(
    tmp_path: Path,
) -> None:
    plan = bias_aware_prospective_download_plan(PROTOCOL)
    for role, object_id, selected_episode_ids in plan.episodes_by_object:
        object_root = tmp_path / "raw" / object_id
        object_root.mkdir(parents=True)
        metadata = {
            "object": RELEASED_METADATA_OBJECT_ALIASES.get(object_id, object_id),
            "sequences": {str(index): {} for index in range(10)},
        }
        (object_root / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        assert role in {"calibration", "target"}
        assert selected_episode_ids

    manifest = build_bias_aware_download_manifest(tmp_path, plan=plan)

    assert manifest["object_count"] == 21
    assert sum(row["role"] == "calibration" for row in manifest["objects"]) == 9
    assert sum(row["role"] == "target" for row in manifest["objects"]) == 12
    aliases = {
        row["object_id"]: row["released_metadata_object"]
        for row in manifest["objects"]
        if row["metadata_identity_alias"]
    }
    assert aliases == RELEASED_METADATA_OBJECT_ALIASES
    assert manifest["information_boundary"]["target_future_opened"] is False
    assert len(manifest["manifest_sha256"]) == 64


def test_download_manifest_rejects_unlisted_metadata_alias(tmp_path: Path) -> None:
    plan = bias_aware_prospective_download_plan(PROTOCOL)
    for _, object_id, _ in plan.episodes_by_object:
        object_root = tmp_path / "raw" / object_id
        object_root.mkdir(parents=True)
        released_object = RELEASED_METADATA_OBJECT_ALIASES.get(object_id, object_id)
        if object_id == "163-bear":
            released_object = "163-bear"
        (object_root / "metadata.json").write_text(
            json.dumps(
                {
                    "object": released_object,
                    "sequences": {str(index): {} for index in range(10)},
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="metadata object changed: 163-bear"):
        build_bias_aware_download_manifest(tmp_path, plan=plan)


def test_by_object_download_avoids_global_tree_and_preserves_allowlist(
    tmp_path: Path,
) -> None:
    plan = bias_aware_prospective_download_plan(PROTOCOL)
    listed_paths: list[str] = []
    downloaded: list[str] = []

    def list_repo_tree(**kwargs: object) -> list[SimpleNamespace]:
        path = str(kwargs["path_in_repo"])
        listed_paths.append(path)
        return [
            SimpleNamespace(blob_id="metadata", path=f"{path}/metadata.json"),
            SimpleNamespace(blob_id="video", path=f"{path}/camera/episode.mp4"),
            SimpleNamespace(blob_id="audio", path=f"{path}/audio.wav"),
            SimpleNamespace(tree_id="camera", path=f"{path}/camera"),
        ]

    def hub_download(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        downloaded.append(filename)
        destination = Path(str(kwargs["local_dir"])) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name == "metadata.json":
            object_id = destination.parent.name
            destination.write_text(
                json.dumps(
                    {
                        "object": RELEASED_METADATA_OBJECT_ALIASES.get(
                            object_id, object_id
                        ),
                        "sequences": {str(index): {} for index in range(10)},
                    }
                ),
                encoding="utf-8",
            )
        else:
            destination.write_bytes(b"video")
        return str(destination)

    manifest = download_bias_aware_prospective_panel_by_object(
        PROTOCOL,
        tmp_path,
        max_workers=2,
        object_delay_seconds=0.0,
        list_repo_tree=list_repo_tree,
        hub_download=hub_download,
    )

    assert listed_paths == [f"raw/{object_id}" for object_id in plan.object_ids]
    assert len(downloaded) == 2 * len(plan.object_ids)
    assert all(path.startswith("raw/") for path in downloaded)
    assert not any(path.endswith(".wav") for path in downloaded)
    assert manifest["object_count"] == 21


def test_any_protocol_mutation_fails_canonical_hash(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["config"]["method"]["minimum_physical_agreement_gain"] = 0.39
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="config hash changed"):
        load_bias_aware_prospective_protocol(mutated)
