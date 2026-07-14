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
    assert breakdown.total_penalty == 0.0
    assert "clip_iqa" in (breakdown.cap_reason or "")


def test_compute_mos_clip_iqa_without_frame_falls_back_to_rule() -> None:
    cfg = ReportConfig(mos_model="clip_iqa")
    mos, breakdown = compute_mos([], cfg, frame_bgr=None)
    assert mos == 4.5
    assert breakdown.cap_reason is None
