import json
from pathlib import Path
from types import SimpleNamespace

from bayesian_phystwin.deform360_bias_aware_prospective_v2_download import (
    bias_aware_prospective_v2_fresh_download_plan,
    build_bias_aware_prospective_v2_download_manifest,
    download_bias_aware_prospective_v2_fresh_by_object,
)


ROOT = Path(__file__).parents[1]
PROTOCOL = (
    ROOT
    / "configs"
    / "sota"
    / "deform360_bias_aware_guarded_belief_prospective_v2.json"
)


def test_v2_plan_contains_only_three_fresh_calibration_objects() -> None:
    plan = bias_aware_prospective_v2_fresh_download_plan(PROTOCOL)

    assert plan.calibration_objects == (
        "078-fishing-line",
        "161-tube",
        "088-snake",
    )
    assert plan.target_objects == ()
    assert plan.object_ids == plan.calibration_objects
    assert plan.episodes_by_object == (
        ("calibration", "078-fishing-line", (4,)),
        ("calibration", "161-tube", (4,)),
        ("calibration", "088-snake", (1,)),
    )
    assert set(plan.allow_patterns) == {
        f"raw/{object_id}/*" for object_id in plan.object_ids
    }


def test_v2_download_manifest_excludes_reserved_targets(tmp_path: Path) -> None:
    plan = bias_aware_prospective_v2_fresh_download_plan(PROTOCOL)
    for _, object_id, _ in plan.episodes_by_object:
        object_root = tmp_path / "raw" / object_id
        object_root.mkdir(parents=True)
        (object_root / "metadata.json").write_text(
            json.dumps(
                {
                    "object": object_id,
                    "sequences": {str(index): {} for index in range(10)},
                }
            ),
            encoding="utf-8",
        )

    manifest = build_bias_aware_prospective_v2_download_manifest(
        tmp_path, plan=plan
    )

    assert manifest["object_count"] == 3
    assert all(row["role"] == "calibration" for row in manifest["objects"])
    assert manifest["information_boundary"]["reserved_target_downloaded"] is False
    assert manifest["information_boundary"]["reserved_target_media_read"] is False


def test_v2_by_object_download_never_enumerates_other_objects(
    tmp_path: Path,
) -> None:
    plan = bias_aware_prospective_v2_fresh_download_plan(PROTOCOL)
    listed: list[str] = []
    downloaded: list[str] = []

    def list_repo_tree(**kwargs: object) -> list[SimpleNamespace]:
        path = str(kwargs["path_in_repo"])
        listed.append(path)
        return [
            SimpleNamespace(blob_id="metadata", path=f"{path}/metadata.json"),
            SimpleNamespace(blob_id="video", path=f"{path}/camera/episode.mp4"),
            SimpleNamespace(blob_id="audio", path=f"{path}/audio.wav"),
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
                        "object": object_id,
                        "sequences": {str(index): {} for index in range(10)},
                    }
                ),
                encoding="utf-8",
            )
        else:
            destination.write_bytes(b"video")
        return str(destination)

    manifest = download_bias_aware_prospective_v2_fresh_by_object(
        PROTOCOL,
        tmp_path,
        max_workers=2,
        object_delay_seconds=0.0,
        list_repo_tree=list_repo_tree,
        hub_download=hub_download,
    )

    assert listed == [f"raw/{object_id}" for object_id in plan.object_ids]
    assert len(downloaded) == 2 * len(plan.object_ids)
    assert all(path.split("/")[1] in plan.object_ids for path in downloaded)
    assert not any(path.endswith(".wav") for path in downloaded)
    assert manifest["object_count"] == 3
