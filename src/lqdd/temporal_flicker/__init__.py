"""TemporalFlicker — 时域闪烁检测器（帧间聚合层）。

本检测器依赖多帧输入，不进入单帧 pipeline 的 ALL_DETECTOR_NAMES，
仅在 VideoClipRunner 的帧间聚合层调用。详见 detector.py。
"""

from lqdd.temporal_flicker.detector import (
    FlickerSegment,
    TemporalFlickerResult,
    detect_temporal_flicker,
)

__all__ = ["FlickerSegment", "TemporalFlickerResult", "detect_temporal_flicker"]
