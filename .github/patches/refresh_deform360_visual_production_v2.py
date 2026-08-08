from __future__ import annotations

from pathlib import Path

SCRIPT = Path("scripts/science/execute_deform360_calibration_visual_production.py")
TESTS_WORKFLOW = Path(".github/workflows/tests.yml")

source = SCRIPT.read_text(encoding="utf-8")
old_preflight = '''    jobs = [cast(Mapping[str, Any], row) for row in admission["jobs"]]
    sources: dict[str, Path] = {}
    for job in jobs:
        job_id = cast(str, job["job_id"])
        sources[job_id] = _verify_source(
            retained_root,
            cast(Mapping[str, Any], job["source_video"]),
            label=f"source video {job_id}",
        )
        _verify_source(
            retained_root,
            cast(Mapping[str, Any], job["source_timestamps"]),
            label=f"source timestamps {job_id}",
        )

    lock_path = run_root / ".production.lock"
'''
new_preflight = '''    jobs = [cast(Mapping[str, Any], row) for row in admission["jobs"]]
    lock_path = run_root / ".production.lock"
'''
if source.count(old_preflight) != 1:
    raise SystemExit("bulk source-preflight block changed")
source = source.replace(old_preflight, new_preflight)

old_loop = '''        rows: list[dict[str, object]] = []
        for job in jobs:
            existing = _existing_receipt(
'''
new_loop = '''        rows: list[dict[str, object]] = []
        for job in jobs:
            job_id = cast(str, job["job_id"])
            source_video = _verify_source(
                retained_root,
                cast(Mapping[str, Any], job["source_video"]),
                label=f"source video {job_id}",
            )
            _verify_source(
                retained_root,
                cast(Mapping[str, Any], job["source_timestamps"]),
                label=f"source timestamps {job_id}",
            )
            existing = _existing_receipt(
'''
if source.count(old_loop) != 1:
    raise SystemExit("locked job-loop block changed")
source = source.replace(old_loop, new_loop)

old_source = '''                source_video_path=sources[cast(str, job["job_id"])],
'''
new_source = '''                source_video_path=source_video,
'''
if source.count(old_source) != 1:
    raise SystemExit("command source-path binding changed")
source = source.replace(old_source, new_source)
SCRIPT.write_text(source, encoding="utf-8")

workflow = TESTS_WORKFLOW.read_text(encoding="utf-8")
anchor = '''            tests/test_deform360_calibration_visual_execution_admission_edges.py \\
            tests/test_deform360_calibration_bundle.py \\
'''
replacement = '''            tests/test_deform360_calibration_visual_execution_admission_edges.py \\
            tests/test_deform360_calibration_visual_production.py \\
            tests/test_deform360_calibration_visual_production_workflow.py \\
            tests/test_deform360_calibration_bundle.py \\
'''
if replacement not in workflow:
    if workflow.count(anchor) != 1:
        raise SystemExit("full-suite visual-production insertion point changed")
    workflow = workflow.replace(anchor, replacement)
TESTS_WORKFLOW.write_text(workflow, encoding="utf-8")
