"""D1 VLM 画质自然语言描述（vlm_caption）单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from lqdd.agent.orchestrator import _execute_vlm_caption
from lqdd.config.loader import AppConfig, VLMConfig
from lqdd.models.agent import AgentContext
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput


class _FakeVLM:
    """Mock VLM client：根据 prompt 关键字返回不同 caption JSON。"""

    def __init__(self, response: dict | None) -> None:
        self.response = response
        self.last_prompt: str | None = None

    def confirm(self, prompt: str, image_b64: str) -> dict | None:
        self.last_prompt = prompt
        return self.response


def _make_ctx(frame=None) -> AgentContext:
    frame = frame if frame is not None else np.full((32, 32, 3), 128, dtype=np.uint8)
    fi = SingleFrameInput(frame=frame, frame_id="f0", mode="fast")
    scan = GlobalScanOutput(
        frame_index=0,
        segmentation_map=np.zeros((32, 32), dtype=np.uint8),
        global_quality_score=0.5,
        is_fast_pass=False,
        is_fast_reject=False,
        nominations=[],
        scan_duration_ms=1.0,
    )
    return AgentContext(frame_input=fi, scan_output=scan)


def _make_config() -> AppConfig:
    # 用最小可用配置；vlm.max_calls_per_frame 默认 3
    from lqdd.config.loader import (
        AgentConfig,
        BandingConfig,
        BackgroundArtifactConfig,
        BlurConfig,
        CompressionConfig,
        EdgeBleedConfig,
        FaceArtifactConfig,
        GlobalScanConfig,
        HairTextureConfig,
        HandAnomalyConfig,
        JudgeConfig,
        MosaicConfig,
        ReportConfig,
    )

    return AppConfig(
        global_scan=GlobalScanConfig(),
        edge_bleed=EdgeBleedConfig(),
        compression=CompressionConfig(),
        blur=BlurConfig(),
        mosaic=MosaicConfig(),
        banding=BandingConfig(),
        background_artifact=BackgroundArtifactConfig(),
        hair_texture=HairTextureConfig(),
        face_artifact=FaceArtifactConfig(),
        hand_anomaly=HandAnomalyConfig(),
        report=ReportConfig(),
        agent=AgentConfig(),
        vlm=VLMConfig(),
        judge=JudgeConfig(),
    )


def test_vlm_caption_success():
    response = {
        "overall_quality": "good",
        "caption": "整体画质良好，左上角存在轻度压缩块效应，对观看体验影响较小。",
        "primary_degradations": ["compression_artifact"],
        "affected_regions": ["左上角"],
        "ux_impact": "对观看体验影响较小",
    }
    ctx = _make_ctx()
    cfg = _make_config()
    vlm = _FakeVLM(response)

    obs = _execute_vlm_caption(ctx, vlm, cfg)

    assert ctx.quality_caption is not None
    assert ctx.quality_caption["overall_quality"] == "good"
    assert "压缩块效应" in ctx.quality_caption["caption"]
    assert ctx.quality_caption["primary_degradations"] == ["compression_artifact"]
    assert ctx.vlm_calls_count == 1
    assert ctx.vlm_ms > 0
    assert "vlm_caption 完成" in obs
    # 应该有一条 trace
    assert any(t.stage == "vlm_caption" for t in ctx.traces)


def test_vlm_caption_skipped_if_already_set():
    ctx = _make_ctx()
    cfg = _make_config()
    ctx.quality_caption = {"overall_quality": "fair", "caption": "已存在"}
    vlm = _FakeVLM({"overall_quality": "good", "caption": "不应被覆盖"})

    obs = _execute_vlm_caption(ctx, vlm, cfg)

    assert "vlm_caption 跳过" in obs
    assert ctx.quality_caption["caption"] == "已存在"
    assert ctx.vlm_calls_count == 0


def test_vlm_caption_skipped_when_calls_exhausted():
    ctx = _make_ctx()
    cfg = _make_config()
    cfg.vlm.max_calls_per_frame = 0  # 立即达上限
    vlm = _FakeVLM({"overall_quality": "good", "caption": "不应被调用"})

    obs = _execute_vlm_caption(ctx, vlm, cfg)

    assert "VLM 调用次数已达上限" in obs
    assert ctx.quality_caption is None
    assert ctx.vlm_calls_count == 0


def test_vlm_caption_handles_vlm_unavailable():
    ctx = _make_ctx()
    cfg = _make_config()
    vlm = _FakeVLM(None)  # VLM 服务不可用

    obs = _execute_vlm_caption(ctx, vlm, cfg)

    assert "VLM 服务不可用" in obs
    assert ctx.quality_caption is None
    # 即使失败也计数 + 记 trace
    assert ctx.vlm_calls_count == 1
    assert any(t.stage == "vlm_caption" and t.decision == "vlm_caption_failed" for t in ctx.traces)


def test_vlm_caption_handles_malformed_response():
    ctx = _make_ctx()
    cfg = _make_config()
    vlm = _FakeVLM({"unrelated_field": "xxx"})  # 缺 caption 字段

    obs = _execute_vlm_caption(ctx, vlm, cfg)

    # 解析成功但字段缺失 -> 用默认空值填充
    assert "vlm_caption 完成" in obs
    assert ctx.quality_caption is not None
    assert ctx.quality_caption["caption"] == ""
    assert ctx.quality_caption["overall_quality"] == "unknown"


def test_vlm_caption_handles_non_dict_response():
    ctx = _make_ctx()
    cfg = _make_config()
    vlm = _FakeVLM("not a json string at all")  # type: ignore[arg-type]

    obs = _execute_vlm_caption(ctx, vlm, cfg)

    # 解析失败 -> 不写入 quality_caption
    assert "解析失败" in obs
    assert ctx.quality_caption is None
