import numpy as np
import pytest

from lqdd.models.inputs import SingleFrameInput
from lqdd.models.report import (
    DegradationItem,
    Evidence,
    PerformanceMetrics,
    QualityReport,
    RootCauseHypothesis,
    TraceEntry,
)
from lqdd.pipeline.video_clip_runner import VideoClipRunner
from lqdd.temporal_flicker import detect_temporal_flicker


def _solid_frame(h: int = 64, w: int = 64, bgr=(120, 120, 120)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bgr
    return img


# ---------------------------------------------------------------------------
# TemporalFlicker 纯算法层
# ---------------------------------------------------------------------------


def test_flicker_stable_frames_no_flicker() -> None:
    frames = [_solid_frame() for _ in range(5)]
    res = detect_temporal_flicker(frames)
    assert res.frame_count == 5
    assert res.flicker_segments == []
    assert res.is_flickering is False
    assert res.flicker_ratio == 0.0


def test_flicker_detects_brightness_jump() -> None:
    # 前 3 帧正常亮度，第 4 帧大幅变亮（注入亮度跳变），第 5 帧恢复
    frames = [_solid_frame(bgr=(120, 120, 120)) for _ in range(3)]
    frames.append(_solid_frame(bgr=(220, 220, 220)))
    frames.append(_solid_frame(bgr=(120, 120, 120)))
    res = detect_temporal_flicker(frames, luma_delta_threshold=8.0)
    assert res.is_flickering is True
    assert res.flicker_ratio > 0.0
    assert res.max_luma_delta > 8.0
    # 至少检出 2 段跳变（第3→4帧、第4→5帧）
    assert len(res.flicker_segments) >= 2
    assert all(s.metric == "luma_delta" for s in res.flicker_segments)


def test_flicker_too_few_frames_returns_empty() -> None:
    res = detect_temporal_flicker([_solid_frame()])
    assert res.frame_count == 1
    assert res.is_flickering is False
    assert res.flicker_segments == []


# ---------------------------------------------------------------------------
# VideoClipRunner 聚合层（用 stub pipeline，不依赖 Ollama）
# ---------------------------------------------------------------------------


def _stub_report(frame_index: int, mos: float, deg_types: list[str]) -> QualityReport:
    degradations = [
        DegradationItem(
            degradation_id=f"deg_{i}",
            region_type="background",
            degradation_type=t,
            severity="minor",
            confidence=0.7,
            bbox=[0, 0, 10, 10],
            frame_indices=[frame_index],
            description="stub",
            detector="stub",
            evidence=Evidence(method="stub", metric="stub", value=1.0, threshold=0.5, detail="stub"),
            root_cause_hypothesis=RootCauseHypothesis(cause="encoding_loss", confidence=0.5),
        )
        for i, t in enumerate(deg_types)
    ]
    return QualityReport(
        report_id=f"r{frame_index}",
        video_id="clip",
        mode="fast",
        frame_index=frame_index,
        report_timestamp="",
        system_version="0.0",
        overall_mos=mos,
        severity="minor",
        degradations=degradations,
        decision_trace=[],
        performance=PerformanceMetrics(total_ms=1.0, global_scan_ms=0.0, detection_ms=0.0, aggregation_ms=0.0),
    )


class _StubPipeline:
    """最小 stub：按帧返回预设 QualityReport，不调用任何外部服务。"""

    def __init__(self, mos_by_frame: list[float], deg_types_by_frame: list[list[str]]) -> None:
        self._mos = mos_by_frame
        self._degs = deg_types_by_frame

    def run(self, frame_input: SingleFrameInput) -> QualityReport:
        idx = frame_input.frame_id.rsplit("_f", 1)[-1]
        i = int(idx)
        return _stub_report(i, self._mos[i], self._degs[i])


def test_video_clip_runner_aggregates_mos_and_degradations() -> None:
    mos = [4.0, 3.5, 4.5]
    degs = [["blockiness"], ["blockiness", "blur"], ["mosaic"]]
    pipeline = _StubPipeline(mos, degs)
    runner = VideoClipRunner(pipeline)
    frames = [_solid_frame() for _ in range(3)]

    result = runner.run(frames, clip_id="c1")
    assert result.clip_id == "c1"
    assert result.frame_count == 3
    assert result.aggregate_mos == pytest.approx(np.mean(mos), rel=1e-3)
    assert result.worst_frame_mos == 3.5
    assert result.worst_frame_index == 1
    # 跨帧 degradation 汇总
    assert result.degradation_summary == {"blockiness": 2, "blur": 1, "mosaic": 1}
    # 稳定帧无闪烁
    assert result.flicker_result.is_flickering is False


def test_video_clip_runner_detects_flicker_across_frames() -> None:
    pipeline = _StubPipeline([4.0, 4.0, 4.0, 4.0], [[] for _ in range(4)])
    runner = VideoClipRunner(pipeline)
    frames = [_solid_frame(bgr=(120, 120, 120)) for _ in range(2)]
    frames.append(_solid_frame(bgr=(230, 230, 230)))  # 亮度跳变
    frames.append(_solid_frame(bgr=(120, 120, 120)))

    result = runner.run(frames, clip_id="c2")
    assert result.flicker_result.is_flickering is True
    assert result.flicker_result.flicker_ratio > 0.0


def test_video_clip_runner_rejects_empty_frames() -> None:
    runner = VideoClipRunner(_StubPipeline([], []))
    with pytest.raises(ValueError):
        runner.run([], clip_id="empty")
