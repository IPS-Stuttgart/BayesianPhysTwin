"""Independent score/hash verification; no raw-data reads or model refits."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read(path):
    return json.loads(path.read_text())


def events(values, thresholds):
    load = values[..., 0] > thresholds[0]
    imbalance = np.abs(values[..., 1]) > thresholds[1]
    horizontal = np.abs(values[..., 2]) > thresholds[2]
    vertical = np.abs(values[..., 3]) > thresholds[3]
    center = np.abs(values[..., 4]) > thresholds[4]
    return np.stack(
        [
            load & imbalance,
            load | imbalance,
            horizontal & vertical,
            horizontal | vertical | center,
            (horizontal & vertical) | (horizontal & center) | (vertical & center),
        ],
        axis=-1,
    )


def verify(root: Path, protocol_path: Path):
    protocol = read(protocol_path)
    result = read(root / "result.json")
    attempt = read(root / "attempt.json")
    assert attempt["launch_count"] == 1
    assert digest(root / "attempt.json") == result["attempt_sha256"]
    assert (
        digest(protocol_path)
        == result["bindings"]["protocols/deform360_joint_events_v1.json"]
    )
    assert attempt["bindings"] == result["bindings"]
    checks = []
    for row in result["objects"]:
        directory = root / row["object_id"]
        source = read(directory / "source_seal.json")
        seal = read(directory / "prediction_seal.json")
        assert digest(directory / "source_fit.npz") == source["source_fit_sha256"]
        assert digest(directory / "source_seal.json") == seal["source_seal_sha256"]
        assert (
            digest(directory / "prediction_seal.json") == row["prediction_seal_sha256"]
        )
        assert digest(directory / "predictions.npz") == seal["predictions_sha256"]
        assert (
            digest(directory / "evaluation_truth.npz") == row["evaluation_truth_sha256"]
        )
        assert source["evaluation_payload_opened"] is False
        assert seal["event_truth_extracted"] is False
        assert (
            datetime.fromisoformat(source["sealed_at"])
            < datetime.fromisoformat(seal["sealed_at"])
            < datetime.fromisoformat(row["scored_at"])
        )
        assert read(directory / "scores.json") == row
        with (
            np.load(directory / "source_fit.npz", allow_pickle=False) as fit,
            np.load(directory / "predictions.npz", allow_pickle=False) as prediction,
            np.load(
                directory / "evaluation_truth.npz", allow_pickle=False
            ) as truth_file,
        ):
            truth = truth_file["query_truth"]
            labels = events(truth, prediction["thresholds"]).astype(float)
            np.testing.assert_allclose(
                prediction["point_field"] @ prediction["query_weights"].T,
                prediction["mean"],
                atol=1e-14,
            )
            marginal = np.sort(fit["_draws_structured_gaussian"], axis=1)
            for arm in protocol["dependence_arms"]:
                samples = fit[f"_draws_{arm}"]
                np.testing.assert_array_equal(np.sort(samples, axis=1), marginal)
                np.testing.assert_allclose(samples.mean(axis=1), 0, atol=1e-12)
                expected = np.stack(
                    [
                        events(mean + samples, prediction["thresholds"])
                        .mean(axis=1)
                        .mean(axis=0)
                        for mean in prediction["mean"]
                    ]
                )
                np.testing.assert_array_equal(prediction[f"p_{arm}"], expected)
            for name in prediction.files:
                if not name.startswith("p_"):
                    continue
                probability = prediction[name]
                assert probability.shape == labels.shape
                act = probability < protocol["loss"]["fallback"]
                stored = row["metrics"][name[2:]]
                np.testing.assert_allclose(
                    np.mean((probability - labels) ** 2),
                    stored["brier"],
                    rtol=0,
                    atol=1e-14,
                )
                np.testing.assert_allclose(
                    np.where(act, labels, 0.1).mean(),
                    stored["decision_loss"],
                    rtol=0,
                    atol=1e-14,
                )
                np.testing.assert_allclose(
                    act.mean(), stored["execute_fraction"], rtol=0, atol=1e-14
                )
            assert row["metrics"]["always_fallback"]["decision_loss"] == 0.1
            np.testing.assert_allclose(
                row["metrics"]["always_execute"]["decision_loss"], labels.mean()
            )
            np.testing.assert_allclose(row["event_frequencies"], labels.mean(axis=0))
            coverage = np.mean(
                (truth >= prediction["lower90"]) & (truth <= prediction["upper90"])
            )
            np.testing.assert_allclose(coverage, row["marginal_coverage90"])
        checks.append({"object_id": row["object_id"], "verified": True})
    for method, values in result["summary"].get("methods", {}).items():
        for metric, value in values.items():
            expected = np.mean(
                [row["metrics"][method][metric] for row in result["objects"]]
            )
            np.testing.assert_allclose(value, expected, rtol=0, atol=1e-14)
    assert len(result["objects"]) == result["summary"]["object_count"]
    return {
        "schema": "deform360-joint-events-independent-verification-v1",
        "verified": True,
        "result_sha256": digest(root / "result.json"),
        "objects": checks,
        "verifier_sha256": digest(Path(__file__)),
        "raw_recordings_read": False,
        "empirical_refits_or_reruns": 0,
        "technical_failure_count": len(result["technical_failures"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.run, args.protocol)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "verified": value["verified"],
                "objects": len(value["objects"]),
                "result_sha256": value["result_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
