import json

import jsonschema
import numpy as np
import pytest

from lqdd.agent.judge_client import MockJudgeClient
from lqdd.config.loader import load_config
from lqdd.models.inputs import SingleFrameInput
from lqdd.models.report import report_to_dict
from lqdd.pipeline.agent_pipeline import AgentPipeline
from lqdd.vlm.client import MockVLMClient

REPO = __import__("pathlib").Path(__file__).resolve().parents[2]


def test_agent_report_v1_schema(schema_v1_path, mock_vlm_fixture, mock_judge_dispatch_fixture) -> None:
    config = load_config(REPO / "config.example.yaml")
    pipeline = AgentPipeline(
        config,
        vlm_client=MockVLMClient(mock_vlm_fixture),
        judge_client=MockJudgeClient(mock_judge_dispatch_fixture),
    )
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    report = pipeline.run(SingleFrameInput(frame=frame, frame_id="v1", mode="fast"))
    data = report_to_dict(report)
    schema = json.loads(schema_v1_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=data, schema=schema)
