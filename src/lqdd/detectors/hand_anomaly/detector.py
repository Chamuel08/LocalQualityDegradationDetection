from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from lqdd.config.loader import HandAnomalyConfig
from lqdd.detectors.base import (
    bbox_from_mask,
    clip_bbox,
    foreground_mask_from_scan,
    hand_region_mask,
    mask_from_heatmap,
)
from lqdd.detectors.helpers import make_degradation_item
from lqdd.models.enums import RegionType, RootCauseCategory, Severity
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput

logger = logging.getLogger(__name__)

# mediapipe 0.10+ Tasks API 所需的手部关键点模型（.task 文件）
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_HAND_MODEL_CACHE: Path = Path.home() / ".cache" / "lqdd" / "hand_landmarker.task"


def _ensure_hand_model() -> Path | None:
    """确保本地存在 hand_landmarker.task，不存在则自动下载（~7MB）。"""
    if _HAND_MODEL_CACHE.exists():
        return _HAND_MODEL_CACHE
    try:
        _HAND_MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading hand_landmarker.task (~7MB) to %s ...", _HAND_MODEL_CACHE)
        urllib.request.urlretrieve(_HAND_MODEL_URL, _HAND_MODEL_CACHE)
        logger.info("Download complete.")
        return _HAND_MODEL_CACHE
    except Exception as exc:
        logger.warning("Failed to download hand_landmarker.task: %s", exc)
        return None


def _mediapipe_hand_score(frame_bgr: np.ndarray) -> tuple[float, str] | None:
    """使用 mediapipe 0.10+ Tasks API（HandLandmarker）计算手部几何异常分数。

    返回 (score, method) 或 None（mediapipe 不可用 / 未检测到手 / 模型文件缺失）。
    score = 各手的指尖到手腕距离的离散度均值，越高表示手部姿态越异常。
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError:
        return None

    model_path = _ensure_hand_model()
    if model_path is None:
        return None

    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    try:
        with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)
    except Exception as exc:
        logger.warning("MediaPipe HandLandmarker inference failed: %s", exc)
        return None

    if not result.hand_landmarks:
        return None

    spreads: list[float] = []
    for hand_landmarks in result.hand_landmarks:
        # landmark 索引与旧版一致：0=wrist, 4/8/12/16/20=指尖
        pts = np.array([(lm.x, lm.y) for lm in hand_landmarks], dtype=np.float32)
        wrist = pts[0]
        tips = pts[[4, 8, 12, 16, 20]]
        dists = np.linalg.norm(tips - wrist, axis=1)
        spreads.append(float(dists.std() / (dists.mean() + 1e-6)))

    score = float(np.mean(spreads))
    return score, "mediapipe_hand_geometry"


def _fallback_hand_edge_score(gray: np.ndarray, hand_mask: np.ndarray) -> tuple[float, str]:
    edges = cv2.Canny(gray, 60, 140)
    edge_density = float(edges[hand_mask].mean() / 255.0) if hand_mask.any() else 0.0
    return edge_density, "hand_edge_density"


class HandAnomalyDetector:
    name = "hand_anomaly"

    def __init__(self, config: HandAnomalyConfig) -> None:
        self.config = config

    def detect(
        self,
        frame_input: SingleFrameInput,
        scan_output: GlobalScanOutput,
    ) -> list:
        frame = frame_input.frame
        h, w = frame.shape[:2]
        fg = foreground_mask_from_scan(scan_output, h, w)
        hand = hand_region_mask(fg, h, w)
        if hand.sum() < self.config.min_hand_pixels:
            return []

        mp_result = _mediapipe_hand_score(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if mp_result is not None:
            value, method = mp_result
            threshold = self.config.finger_spread_threshold
            triggered = value >= threshold
            metric = "finger_spread_ratio"
            detail = f"MediaPipe 手指伸展离散度 {value:.2f} ≥ {threshold}"
        else:
            value, method = _fallback_hand_edge_score(gray, hand)
            threshold = self.config.edge_density_threshold
            triggered = value >= threshold
            metric = "hand_edge_density"
            detail = f"手部 ROI 边缘密度 {value:.2f} ≥ {threshold}（无 MediaPipe 时的几何代理）"

        if not triggered:
            return []

        heat = np.zeros((h, w), dtype=np.float32)
        heat[hand] = value
        region_mask = mask_from_heatmap(heat, threshold=threshold * 0.85, roi_mask=hand, min_area=150)
        bbox = clip_bbox(bbox_from_mask(region_mask) if region_mask is not None else bbox_from_mask(hand), w, h)

        return [
            make_degradation_item(
                detector=self.name,
                degradation_type="hand_anomaly",
                region_type=RegionType.HAND,
                severity=Severity.MINOR.value,
                confidence=0.66,
                bbox=bbox,
                region_mask=region_mask,
                method=method,
                metric=metric,
                value=value,
                threshold=threshold,
                detail=detail,
                description="手部区域几何/边缘异常（多指、扭曲等代理信号）",
                cause=RootCauseCategory.GENERATION_ARTIFACT,
                frame_index=scan_output.frame_index,
            )
        ]