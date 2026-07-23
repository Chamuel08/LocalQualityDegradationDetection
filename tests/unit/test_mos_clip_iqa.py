"""CLIP-IQA MOS 后端测试。

pyiqa / torch 未安装时全部跳过（CLIP-IQA 是可选依赖）。
"""
from __future__ import annotations

import numpy as np
import pytest

from lqdd.config.loader import ReportConfig
from lqdd.report.generator import compute_mos

pyiqa = pytest.importorskip("pyiqa")
torch = pytest.importorskip("torch")


def _fake_frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


def test_clip_iqa_predict_returns_mos_in_range() -> None:
    from lqdd.mos.clip_iqa import predict_mos_clip_iqa

    mos = predict_mos_clip_iqa(_fake_frame(), device="cpu")
    assert 1.0 <= mos <= 5.0


def test_clip_iqa_metric_is_cached() -> None:
    from lqdd.mos import clip_iqa as mod

    mod._CLIP_IQA_METRIC = None  # reset singleton
    m1 = mod._get_metric("cpu")
    m2 = mod._get_metric("cpu")
    assert m1 is m2


def test_compute_mos_clip_iqa_branch() -> None:
    cfg = ReportConfig(mos_model="clip_iqa")
    mos, breakdown = compute_mos([], cfg, frame_bgr=_fake_frame())
    assert 1.0 <= mos <= 5.0
    assert breakdown.status == "ok"
    assert breakdown.model == "clip_iqa"
    assert breakdown.mos == mos
    assert breakdown.reason is None


def test_compute_mos_clip_iqa_without_frame_returns_null() -> None:
    """未提供 frame_bgr → MOS=null + unavailable（不回退默认分）。"""
    cfg = ReportConfig(mos_model="clip_iqa")
    mos, breakdown = compute_mos([], cfg, frame_bgr=None)
    assert mos is None
    assert breakdown.status == "unavailable"
    assert breakdown.reason and "frame_bgr" in breakdown.reason
