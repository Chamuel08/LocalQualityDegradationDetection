import json

import jsonschema


def test_vlm_fixture_validates(vlm_schema_path, mock_vlm_fixture) -> None:
    schema = json.loads(vlm_schema_path.read_text(encoding="utf-8"))
    data = json.loads(mock_vlm_fixture.read_text(encoding="utf-8"))
    jsonschema.validate(instance=data, schema=schema)


def test_judge_fixture_validates(judge_schema_path, mock_judge_dispatch_fixture) -> None:
    schema = json.loads(judge_schema_path.read_text(encoding="utf-8"))
    data = json.loads(mock_judge_dispatch_fixture.read_text(encoding="utf-8"))
    jsonschema.validate(instance=data, schema=schema)
