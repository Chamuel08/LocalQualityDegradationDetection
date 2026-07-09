import json

import jsonschema


def test_judge_dispatch_fixture(judge_schema_path, mock_judge_dispatch_fixture) -> None:
    schema = json.loads(judge_schema_path.read_text(encoding="utf-8"))
    data = json.loads(mock_judge_dispatch_fixture.read_text(encoding="utf-8"))
    jsonschema.validate(instance=data, schema=schema)
