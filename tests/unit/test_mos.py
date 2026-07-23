import pytest

from lqdd.config.loader import ReportConfig
from lqdd.report.generator import compute_mos


def test_compute_mos_no_frame_returns_none_with_reason() -> None:
    """clip_iqa 默认后端：未提供 frame_bgr → MOS=null + unavailable 原因（不回退默认分）。"""
    cfg = ReportConfig()
    mos, breakdown = compute_mos([], cfg, frame_bgr=None)
    assert mos is None
    assert breakdown.status == "unavailable"
    assert breakdown.model == "clip_iqa"
    assert breakdown.reason and "frame_bgr" in breakdown.reason


def test_compute_mos_clip_iqa_unavailable_when_dependency_missing(monkeypatch) -> None:
    """pyiqa 未安装时（模拟 ImportError）→ MOS=null + 原因含安装提示，不回退默认分。"""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("lqdd.mos.clip_iqa"):
            raise ImportError("simulated: pyiqa not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    import numpy as np

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    mos, breakdown = compute_mos([], ReportConfig(), frame_bgr=frame)
    assert mos is None
    assert breakdown.status == "unavailable"
    assert "lqdd[clip_iqa]" in (breakdown.reason or "")


def test_internal_backend_unimplemented_returns_none() -> None:
    cfg = ReportConfig(mos_model="internal")
    mos, breakdown = compute_mos([], cfg, frame_bgr=None)
    assert mos is None
    assert breakdown.status == "unavailable"
    assert breakdown.model == "internal"
