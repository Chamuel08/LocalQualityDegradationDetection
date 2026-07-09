import json
from pathlib import Path

import jsonschema
import pytest


def test_schema_file_exists(schema_path: Path) -> None:
    assert schema_path.is_file()


def test_minimal_report_validates(schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    sample = {
        "report_id": "rpt_test",
        "video_id": "frame_01",
        "mode": "fast",
        "frame_index": 0,
        "report_timestamp": "2026-07-09T12:00:00+00:00",
        "system_version": "0.1.0",
        "overall_mos": 4.3,
        "severity": "minor",
        "degradations": [],
        "decision_trace": [
            {
                "stage": "mode_select",
                "module": "FastPipeline",
                "timestamp_ms": 0.0,
                "duration_ms": 0.0,
                "input_summary": {},
                "output_summary": {},
                "decision": "mode_fast",
                "mode": "fast",
            }
        ],
        "performance": {
            "total_ms": 10.0,
            "global_scan_ms": 5.0,
            "detection_ms": 3.0,
            "aggregation_ms": 2.0,
        },
        "vlm_reasoning_summary": None,
    }
def test_golden_samples_validate(schema_path: Path, expected_dir: Path) -> None:
    if not expected_dir.is_dir():
        pytest.skip("golden expected dir not generated yet")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    files = sorted(expected_dir.glob("*.json"))
    assert files, "no golden JSON files"
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=data, schema=schema)
