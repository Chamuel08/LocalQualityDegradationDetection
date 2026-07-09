import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from lqdd.agent.judge_client import MockJudgeClient
from lqdd.config.loader import load_config
from lqdd.models.inputs import SingleFrameInput
from lqdd.pipeline.agent_pipeline import AgentPipeline
from lqdd.vlm.client import MockVLMClient


REPO = Path(__file__).resolve().parents[2]


def _grey_edge_frame() -> np.ndarray:
    import cv2

    h, w = 120, 160
    img = np.full((h, w, 3), (40, 80, 120), dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), 30, (200, 180, 160), -1)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)
    edge = (dist >= 26) & (dist <= 34)
    img[edge] = (30, 180, 30)
    return img


def test_agent_pipeline_mock_vlm_and_judge(mock_vlm_fixture: Path, mock_judge_dispatch_fixture: Path) -> None:
    config = load_config(REPO / "config.example.yaml")
    vlm = MockVLMClient(mock_vlm_fixture)
    judge = MockJudgeClient(mock_judge_dispatch_fixture)
    pipeline = AgentPipeline(config, vlm_client=vlm, judge_client=judge)
    frame = _grey_edge_frame()
    report = pipeline.run(SingleFrameInput(frame=frame, frame_id="grey_test", mode="fast"))

    stages = {t.stage for t in report.decision_trace}
    assert "routing" in stages
    assert "judge" in stages
    assert report.agent_meta is not None


def test_agent_trace_stages(mock_vlm_fixture: Path) -> None:
    config = load_config(REPO / "config.example.yaml")
    judge = MockJudgeClient(REPO / "tests/fixtures/mock_judge_responses/consistent.json")
    pipeline = AgentPipeline(config, vlm_client=MockVLMClient(mock_vlm_fixture), judge_client=judge)
    report = pipeline.run(SingleFrameInput(frame=_grey_edge_frame(), frame_id="t", mode="fast"))
    stages = [t.stage for t in report.decision_trace]
    assert "aggregation" in stages
    assert "detection" in stages
