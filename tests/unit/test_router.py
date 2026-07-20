import json

import jsonschema
import pytest

from lqdd.config.loader import AgentConfig
from lqdd.agent.router import _confidence_band, is_grey_confidence, route_nominations
from lqdd.models.agent import AgentContext
from lqdd.models.inputs import GlobalScanOutput, RegionNomination


def test_confidence_bands() -> None:
    cfg = AgentConfig()
    assert _confidence_band(0.85, cfg) == "high"
    assert _confidence_band(0.55, cfg) == "grey"
    assert _confidence_band(0.2, cfg) == "low"
    assert is_grey_confidence(0.55, cfg)


def test_route_dispatches_detectors() -> None:
    cfg = AgentConfig()
    scan = GlobalScanOutput(
        frame_index=0,
        segmentation_map=__import__("numpy").zeros((10, 10), dtype=__import__("numpy").uint8),
        global_quality_score=0.5,
        is_fast_pass=False,
        is_fast_reject=False,
        nominations=[
            RegionNomination(
                region_type=4,
                bbox=(0, 0, 10, 10),
                mask=__import__("numpy").ones((10, 10), dtype=bool),
                anomaly_score=0.6,
                confidence=0.55,
                suggested_detectors=["edge_bleed"],
            )
        ],
        scan_duration_ms=1.0,
    )
    ctx = AgentContext()
    dispatched = route_nominations(scan, cfg, ctx)
    assert "edge_bleed" in dispatched
    # ReAct 架构下路由器不再产出 vlm_pending；灰区项统一标记为 dispatch，
    # 是否调用 VLM 由 Agent 在后续步骤自主决定。
    grey = [d for d in ctx.routing_decisions if d.confidence_band == "grey"]
    assert grey and all(d.decision == "dispatch" for d in grey)
