from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "science"
    / "audit_deform360_existing_data_eligibility_v1.py"
)
SPEC = importlib.util.spec_from_file_location("_deform360_existing_data_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

AuditError = MODULE.AuditError
audit = MODULE.audit
load_protocol = MODULE.load_protocol
write_result = MODULE.write_result


def _pair(directory: Path, stem: str, suffix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}{suffix}").write_bytes(b"not opened by audit")
    (directory / f"{stem}.txt").write_text("not opened by audit\n", encoding="utf-8")


def _raw_object(
    root: Path, object_id: str, episodes: int, cameras: int, tactile: int
) -> None:
    object_dir = root / object_id
    stems = [f"capture_{index:02d}" for index in range(episodes)]
    for camera in range(cameras):
        stream = object_dir / f"brics-odroid-{camera:02d}_cam0"
        for stem in stems:
            _pair(stream, stem, ".mp4")
    for sensor in range(tactile):
        stream = object_dir / f"brics-odroid_tactile{sensor}"
        for stem in stems:
            _pair(stream, stem, ".npy")
        (stream / "median_reference.npy").write_bytes(b"ignored")


def _processed_object(root: Path, object_id: str, episodes: int, cameras: int) -> None:
    object_dir = root / object_id
    for episode in range(episodes):
        episode_dir = object_dir / f"episode_{episode:04d}"
        rgb = episode_dir / "rgb"
        rgb.mkdir(parents=True, exist_ok=True)
        for camera in range(cameras):
            (rgb / f"camera_{camera:02d}.mp4").write_bytes(b"video")
        (episode_dir / "robot" / "robot.npz").parent.mkdir(parents=True)
        (episode_dir / "robot" / "robot.npz").write_bytes(b"numeric-not-loaded")
        (episode_dir / "pcd_clean" / "trajectory.npz").parent.mkdir(parents=True)
        (episode_dir / "pcd_clean" / "trajectory.npz").write_bytes(
            b"numeric-not-loaded"
        )
        (episode_dir / "episode.json").write_text(
            json.dumps({"action_primitive": f"primitive-{episode}"}) + "\n",
            encoding="utf-8",
        )


def _protocol(raw_root: Path, processed_root: Path) -> dict[str, object]:
    return {
        "schema": "bayesian-phystwin/deform360-existing-data-eligibility-protocol",
        "schema_version": 1,
        "protocol_id": "fixture",
        "status": "frozen-before-fragment-payload-audit",
        "official_processing_revision": "d" * 40,
        "information_boundary": {
            "directory_and_filename_inventory_only": True,
            "small_metadata_json_allowed": True,
            "media_payload_decoded": False,
            "numeric_arrays_loaded": False,
            "large_payloads_hashed": False,
            "target_outcomes_used": False,
        },
        "metadata": {
            "maximum_json_bytes": 1048576,
            "allowed_basenames": [
                "action.json",
                "config.json",
                "episode.json",
                "info.json",
                "manifest.json",
                "metadata.json",
                "recording.json",
                "task.json",
            ],
            "action_key_fragments": [
                "action",
                "interaction",
                "manipulation",
                "motion",
                "primitive",
                "task",
            ],
        },
        "thresholds": {
            "minimum_raw_camera_pairs_per_episode": 32,
            "minimum_raw_tactile_pairs_per_episode": 4,
            "minimum_processed_camera_videos_per_episode": 24,
            "recommended_common_camera_floor": 32,
            "minimum_episodes_per_transport_object": 2,
            "minimum_multi_episode_objects_for_pilot": 1,
            "minimum_multi_episode_objects_for_bounded_study": 2,
        },
        "runners": {
            "gpuserver6000": {
                "roots": [
                    {
                        "root": str(raw_root),
                        "kind": "raw",
                        "expected_object_ids": ["001-rope"],
                    }
                ]
            },
            "gpuserver4090": {
                "roots": [
                    {
                        "root": str(processed_root),
                        "kind": "processed",
                        "expected_object_ids": ["004-rubber-band"],
                    }
                ]
            },
        },
    }


class Deform360ExistingDataEligibilityTests(unittest.TestCase):
    def test_checked_in_protocol_is_strict_and_has_27_unique_objects(self) -> None:
        protocol_path = (
            Path(__file__).resolve().parents[1]
            / "protocols"
            / "deform360_existing_data_eligibility_v1.json"
        )
        protocol = load_protocol(protocol_path)
        ids = [
            object_id
            for runner in protocol["runners"].values()
            for root in runner["roots"]
            for object_id in root["expected_object_ids"]
        ]
        self.assertEqual(len(ids), 27)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            protocol["official_processing_revision"],
            "d8522a4403b766aeb387510c04e89032a56fdf35",
        )

    def test_raw_multiepisode_visuotactile_candidate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "raw"
            processed_root = root / "processed"
            processed_root.mkdir()
            _raw_object(raw_root, "001-rope", episodes=3, cameras=33, tactile=4)
            result = audit(_protocol(raw_root, processed_root), "gpuserver6000", "a")
            row = result["objects"][0]
            self.assertEqual(
                row["classification"], "raw_visuotactile_transport_candidate"
            )
            self.assertEqual(row["raw_usable_episode_count"], 3)
            self.assertEqual(row["raw_visuotactile_episode_count"], 3)
            self.assertEqual(result["decision"], "sufficient_for_pilot_only")
            self.assertFalse(result["information_boundary"]["numeric_arrays_loaded"])
            self.assertFalse(result["information_boundary"]["media_payload_decoded"])

    def test_processed_candidate_requires_action_and_target_carriers(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "raw"
            processed_root = root / "processed"
            raw_root.mkdir()
            _processed_object(processed_root, "004-rubber-band", episodes=3, cameras=32)
            result = audit(_protocol(raw_root, processed_root), "gpuserver4090", "a")
            row = result["objects"][0]
            self.assertEqual(row["classification"], "processed_transport_ready")
            self.assertEqual(row["processed_transport_ready_episode_count"], 3)
            self.assertTrue(row["action_diversity_verified"])
            self.assertEqual(result["decision"], "sufficient_for_pilot_only")

    def test_single_episode_is_only_calibration_or_control(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "raw"
            processed_root = root / "processed"
            processed_root.mkdir()
            _raw_object(raw_root, "001-rope", episodes=1, cameras=41, tactile=4)
            result = audit(_protocol(raw_root, processed_root), "gpuserver6000", None)
            self.assertEqual(
                result["objects"][0]["classification"],
                "single_episode_calibration_or_control",
            )
            self.assertEqual(
                result["decision"], "insufficient_for_cross_episode_transport"
            )

    def test_result_id_excludes_repository_revision_and_output_is_canonical(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "raw"
            processed_root = root / "processed"
            processed_root.mkdir()
            _raw_object(raw_root, "001-rope", episodes=2, cameras=32, tactile=4)
            protocol = _protocol(raw_root, processed_root)
            first = audit(protocol, "gpuserver6000", "a")
            second = audit(protocol, "gpuserver6000", "b")
            self.assertEqual(first["result_id"], second["result_id"])
            destination = root / "result.json"
            write_result(destination, first)
            self.assertTrue(destination.read_text(encoding="utf-8").endswith("\n"))
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8"))["result_id"],
                first["result_id"],
            )

    def test_duplicate_protocol_key_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text('{"schema": "a", "schema": "b"}\n', encoding="utf-8")
            with self.assertRaises(AuditError):
                load_protocol(path)


if __name__ == "__main__":
    unittest.main()
