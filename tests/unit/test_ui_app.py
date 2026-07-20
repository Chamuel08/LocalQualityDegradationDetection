"""lqdd.ui.app 回调的单元测试。

不启动真实 Gradio server，只测 run_single / run_video 回调逻辑。
通过 monkeypatch 替换 pipeline.run 与 viz，断言返回结构正确。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lqdd.ui import app as ui_app


@pytest.fixture()
def fake_frame() -> np.ndarray:
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture()
def fake_report():
    """构造一个最小 QualityReport 替身，含 agent_meta。"""
    ev = SimpleNamespace(method="m", metric="lap_var", value=10.0, threshold=85.0, detail="偏糊", attention_map=None)
    deg = SimpleNamespace(
        degradation_id="d1",
        region_type=0,
        degradation_type="face_blur",
        severity="minor",
        confidence=0.782,
        mos_impact=-0.2,
        bbox=[10, 10, 20, 20],
        frame_indices=[0],
        description="face blur",
        detector="face_artifact",
        evidence=ev,
        root_cause_hypothesis=SimpleNamespace(cause="generation_artifact", confidence=0.45),
        region_mask_rle=None,
        vlm_reasoning=None,
    )
    perf = SimpleNamespace(
        total_ms=100.0,
        global_scan_ms=10.0,
        detection_ms=20.0,
        aggregation_ms=5.0,
        vlm_ms=50.0,
        judge_ms=15.0,
    )
    return SimpleNamespace(
        report_id="r1",
        video_id="v1",
        mode="fast",
        frame_index=0,
        report_timestamp="t",
        system_version="1.0.0",
        overall_mos=3.673,
        severity="moderate",
        degradations=[deg],
        decision_trace=[],
        performance=perf,
        mos_breakdown=None,
        degradation_summary=None,
        vlm_reasoning_summary=None,
        agent_meta={
            "rounds_executed": 1,
            "vlm_calls": 1,
            "agent_driven_vlm": True,
            "agent_steps": [
                {
                    "step": 1,
                    "action": "vlm_analyze",
                    "thought": "低置信度需确认",
                    "observation": "已确认",
                    "latency_ms": 9450.2,
                }
            ],
            "vlm_discover_findings": [
                {
                    "degradation_type": "hand_extra_finger",
                    "region_description": "左手6指",
                    "severity": "moderate",
                    "confidence": 0.82,
                    "reasoning": "多余手指",
                    "mos_impact_estimate": -0.4,
                }
            ],
        },
    )


def _report_to_dict(report):
    """测试用：把 SimpleNamespace fake_report 转成 dict（绕过 asdict）。"""
    return {
        "report_id": report.report_id,
        "system_version": report.system_version,
        "overall_mos": round(float(report.overall_mos), 3),
        "severity": report.severity,
        "degradations": [
            {
                "degradation_type": d.degradation_type,
                "severity": d.severity,
                "confidence": d.confidence,
                "detector": d.detector,
                "bbox": d.bbox,
            }
            for d in report.degradations
        ],
        "agent_meta": report.agent_meta,
    }


def test_run_single_returns_expected_structure(tmp_path, monkeypatch, fake_frame, fake_report):
    img_path = tmp_path / "frame.png"
    import cv2

    cv2.imwrite(str(img_path), fake_frame)

    # 替换 pipeline 构造与 run
    class _FakePipeline:
        def __init__(self, *a, **kw):
            pass

        def run(self, fi):
            return fake_report

    monkeypatch.setattr(ui_app, "_build_pipeline", lambda cfg, use_agent: _FakePipeline())
    monkeypatch.setattr(ui_app, "_resolve_config", lambda cp: SimpleNamespace(agent=SimpleNamespace(enabled=True), report=SimpleNamespace(system_version="1.0.0")))
    # render_mask_overlay 直接返回原图
    monkeypatch.setattr("lqdd.report.viz_variants.render_mask_overlay", lambda frame, report, style="contour_fill": frame)
    monkeypatch.setattr("lqdd.models.report.report_to_dict", _report_to_dict)

    overlay, summary, deg_rows, agent_rows, vlm_md, full_json = ui_app.run_single(
        str(img_path), "V1 ReAct Agent", ""
    )

    assert overlay.shape == (64, 64, 3)
    assert "MOS" in summary and "3.673" in summary
    assert len(deg_rows) == 1
    assert deg_rows[0][0] == "face_blur"
    assert deg_rows[0][2] == 0.782
    assert len(agent_rows) == 1
    assert agent_rows[0][1] == "vlm_analyze"
    assert "hand_extra_finger" in vlm_md
    assert full_json["overall_mos"] == 3.673
    assert full_json["agent_meta"]["agent_driven_vlm"] is True


def test_run_single_no_image_raises():
    with pytest.raises(ValueError):
        ui_app.run_single("", "V1 ReAct Agent", "")


def test_run_video_returns_expected_structure(tmp_path, monkeypatch, fake_report):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")  # 占位，sample 会被 monkeypatch 掉

    frames = [np.full((32, 32, 3), 100, dtype=np.uint8) for _ in range(3)]

    def _fake_sample(path, max_frames=8, start_sec=0.0, end_sec=None):
        return frames

    flicker = SimpleNamespace(
        frame_count=3,
        flicker_segments=[],
        mean_luma_delta=1.0,
        max_luma_delta=2.0,
        flicker_ratio=0.0,
        is_flickering=False,
        method="temporal_luma_hue_delta",
    )
    clip_report = SimpleNamespace(
        clip_id="clip",
        frame_count=3,
        frame_reports=[fake_report, fake_report, fake_report],
        flicker_result=flicker,
        aggregate_mos=3.5,
        worst_frame_mos=3.5,
        worst_frame_index=0,
        degradation_summary={"face_blur": 3},
    )

    monkeypatch.setattr("lqdd.pipeline.video_clip_runner.sample_frames_from_video", _fake_sample)
    monkeypatch.setattr(
        "lqdd.pipeline.video_clip_runner.VideoClipRunner",
        lambda pipeline, **kw: SimpleNamespace(run=lambda frames, clip_id="clip", frame_ids=None: clip_report),
    )
    monkeypatch.setattr(ui_app, "_build_pipeline", lambda cfg, use_agent: object())
    monkeypatch.setattr(ui_app, "_resolve_config", lambda cp: SimpleNamespace(agent=SimpleNamespace(enabled=True), report=SimpleNamespace(system_version="1.0.0")))
    monkeypatch.setattr("lqdd.models.report.report_to_dict", _report_to_dict)

    summary, summary_rows, full_json = ui_app.run_video(str(video_path), "V1 ReAct Agent", "", 3)

    assert "aggregate_mos" in summary and "3.500" in summary
    assert summary_rows == [["face_blur", 3]]
    assert full_json["frame_count"] == 3
    assert full_json["flicker"]["is_flickering"] is False
    assert len(full_json["frame_reports"]) == 3


def test_run_video_no_video_raises():
    with pytest.raises(ValueError):
        ui_app.run_video("", "V1 ReAct Agent", "", 4)


def test_bundled_config_path_none_when_not_frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert ui_app._bundled_config_path() is None
