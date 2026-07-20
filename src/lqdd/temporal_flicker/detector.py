from __future__ import annotations

"""TemporalFlicker — 时域闪烁检测器。

**设计说明**

本检测器不进入 ALL_DETECTOR_NAMES 和 build_detector_registry，
因为它依赖多帧输入，无法在单帧 pipeline 中被调用。
正确的调用方式是通过 VideoClipRunner 在帧间聚合层使用。

算法：
1. 计算相邻帧的灰度均值差（亮度跳变）
2. 计算相邻帧的 HSV-V 通道均值差（亮度稳定性）
3. 计算相邻帧的色相 H 通道均值差（色彩漂移）
4. 以上三路信号任一超过阈值，即认为该帧间存在闪烁
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FlickerSegment:
    """一段闪烁区间的描述。"""

    start_frame: int
    end_frame: int
    max_delta: float
    metric: str          # "luma_delta" | "hsv_luma_delta" | "hue_delta"
    severity: str        # "minor" | "moderate" | "critical"


@dataclass
class TemporalFlickerResult:
    """对一组连续帧的时域闪烁检测结果。"""

    frame_count: int
    flicker_segments: list[FlickerSegment]
    mean_luma_delta: float
    max_luma_delta: float
    flicker_ratio: float          # 发生闪烁的帧间比例 [0, 1]
    is_flickering: bool
    method: str = "temporal_luma_hue_delta"


def detect_temporal_flicker(
    frames: list[np.ndarray],
    luma_delta_threshold: float = 8.0,
    hue_delta_threshold: float = 6.0,
    min_flicker_ratio: float = 0.15,
) -> TemporalFlickerResult:
    """对一组连续 BGR 帧检测时域闪烁。

    Args:
        frames: 连续帧列表（BGR uint8），至少 2 帧。
        luma_delta_threshold: 相邻帧灰度均值差超过此值视为亮度跳变（范围 0-255）。
        hue_delta_threshold: 相邻帧 H 通道均值差超过此值视为色彩跳变（范围 0-180）。
        min_flicker_ratio: 闪烁帧间比例超过此值才触发整体 is_flickering=True。

    Returns:
        TemporalFlickerResult
    """
    n = len(frames)
    if n < 2:
        return TemporalFlickerResult(
            frame_count=n,
            flicker_segments=[],
            mean_luma_delta=0.0,
            max_luma_delta=0.0,
            flicker_ratio=0.0,
            is_flickering=False,
        )

    luma_means: list[float] = []
    hue_means: list[float] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luma_means.append(float(gray.mean()))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_means.append(float(hsv[:, :, 0].mean()))

    luma_deltas = [abs(luma_means[i + 1] - luma_means[i]) for i in range(n - 1)]
    hue_deltas = [abs(hue_means[i + 1] - hue_means[i]) for i in range(n - 1)]

    flicker_segments: list[FlickerSegment] = []
    flicker_frame_count = 0

    for i, (ld, hd) in enumerate(zip(luma_deltas, hue_deltas)):
        triggered = False
        metric = ""
        delta = 0.0

        if ld >= luma_delta_threshold:
            triggered = True
            metric = "luma_delta"
            delta = ld
        elif hd >= hue_delta_threshold:
            triggered = True
            metric = "hue_delta"
            delta = hd

        if triggered:
            flicker_frame_count += 1
            # 严重度分级
            if metric == "luma_delta":
                severity = "critical" if delta >= luma_delta_threshold * 2.5 else (
                    "moderate" if delta >= luma_delta_threshold * 1.5 else "minor"
                )
            else:
                severity = "critical" if delta >= hue_delta_threshold * 2.5 else (
                    "moderate" if delta >= hue_delta_threshold * 1.5 else "minor"
                )
            flicker_segments.append(
                FlickerSegment(
                    start_frame=i,
                    end_frame=i + 1,
                    max_delta=round(delta, 3),
                    metric=metric,
                    severity=severity,
                )
            )

    mean_ld = float(np.mean(luma_deltas)) if luma_deltas else 0.0
    max_ld = float(max(luma_deltas)) if luma_deltas else 0.0
    flicker_ratio = flicker_frame_count / (n - 1) if n > 1 else 0.0

    return TemporalFlickerResult(
        frame_count=n,
        flicker_segments=flicker_segments,
        mean_luma_delta=round(mean_ld, 3),
        max_luma_delta=round(max_ld, 3),
        flicker_ratio=round(flicker_ratio, 3),
        is_flickering=(flicker_ratio >= min_flicker_ratio),
    )