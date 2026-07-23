"""TemporalFlicker 时序建模升级（C1 运动补偿 / C2 时序 SSIM / C3 局部闪烁）单元测试。"""
from __future__ import annotations

import cv2
import numpy as np

from lqdd.temporal_flicker.detector import detect_temporal_flicker


def _solid(h, w, val):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _stable_frames(n=5, h=64, w=64):
    return [_solid(h, w, 120) for _ in range(n)]


def test_stable_frames_no_flicker():
    res = detect_temporal_flicker(_stable_frames())
    assert res.is_flickering is False
    assert res.flicker_segments == []
    assert res.mean_luma_delta == 0.0
    # C1: 运动补偿残差接近 0
    assert res.mean_motion_compensated_delta < 1.0
    # C2: 时序 SSIM 接近 1
    assert res.temporal_ssim > 0.95


def test_single_frame_returns_empty():
    res = detect_temporal_flicker([_solid(32, 32, 100)])
    assert res.is_flickering is False
    assert res.frame_count == 1


def test_luma_flicker_detected():
    frames = [
        _solid(64, 64, 100),
        _solid(64, 64, 100),
        _solid(64, 64, 200),  # 亮度跳变
        _solid(64, 64, 100),
        _solid(64, 64, 100),
    ]
    res = detect_temporal_flicker(frames)
    assert res.is_flickering is True
    assert res.flicker_ratio > 0
    assert res.max_luma_delta >= 90.0
    # 至少有一个 luma_delta 段
    assert any(s.metric == "luma_delta" for s in res.flicker_segments)


def test_motion_compensation_distinguishes_motion_from_flicker():
    """纹理图平移（无闪烁）：luma_delta 小，运动补偿后残差也小，时序 SSIM 高。

    用高斯平滑后的随机噪声作为纹理图（光流有足够纹理可跟踪），
    通过 cv2.warpAffine 平移 5px（BORDER_REPLICATE 避免边界不连续）。
    """
    h, w = 64, 64
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    # 高斯模糊生成平滑纹理：光流可跟踪，又不会因纯噪声让残差过大
    base_gray = cv2.GaussianBlur(noise, (15, 15), 0)
    base = np.stack([base_gray, base_gray, base_gray], axis=-1).astype(np.uint8)
    # 平移 5px（BORDER_REPLICATE 避免边界不连续）
    M = np.float32([[1, 0, 5], [0, 1, 0]])
    shifted = cv2.warpAffine(base, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    frames = [base, shifted]
    res = detect_temporal_flicker(frames, enable_motion_compensation=True)
    # 平移不改变全局均值（BORDER_REPLICATE 仅边界少量像素）-> luma_delta 小
    assert res.mean_luma_delta < 5.0
    # 运动补偿后残差应较小（光流对齐了平移）
    assert res.mean_motion_compensated_delta < 15.0
    # 时序 SSIM 经运动补偿后应较高
    assert res.temporal_ssim > 0.8


def test_motion_compensation_can_be_disabled():
    frames = _stable_frames()
    res = detect_temporal_flicker(frames, enable_motion_compensation=False)
    assert res.mean_motion_compensated_delta == 0.0
    assert res.temporal_ssim == 1.0
    assert res.flicker_heatmap is None


def test_localized_heatmap_and_segments():
    """局部区域闪烁应产出热力图 + 带 bbox 的局部段。"""
    h, w = 64, 64
    frames = []
    for i in range(6):
        f = np.full((h, w, 3), 100, dtype=np.uint8)
        if i % 2 == 1:
            # 仅左上角 32x32 闪烁
            f[:32, :32] = 200
        frames.append(f)
    res = detect_temporal_flicker(frames, block_size=16, motion_delta_threshold=10.0)
    assert res.flicker_heatmap is not None
    assert res.flicker_heatmap.shape == (h, w)
    # 热力图左上角应明显高于右下角
    assert res.flicker_heatmap[:32, :32].mean() > res.flicker_heatmap[32:, 32:].mean()
    # 应至少有一个带 bbox 的局部段，且 bbox 落在左上区域
    assert len(res.localized_segments) >= 1
    seg = res.localized_segments[0]
    assert seg.bbox is not None
    assert seg.bbox[0] < 40 and seg.bbox[1] < 40


def test_heatmap_disabled_when_block_size_zero():
    frames = _stable_frames()
    res = detect_temporal_flicker(frames, block_size=0)
    assert res.flicker_heatmap is None
    assert res.localized_segments == []


def test_method_string_updated():
    res = detect_temporal_flicker(_stable_frames())
    assert "motion" in res.method or "ssim" in res.method
