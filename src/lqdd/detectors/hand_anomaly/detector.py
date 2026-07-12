from __future__ import annotations

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


def _mediapipe_hand_score(frame_bgr: np.ndarray) -> tuple[float, str] | None:
    try:
        import mediapipe as mp
    except ImportError:
        return None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.4,
    ) as hands:
        result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None

    spreads: list[float] = []
    for hand_landmarks in result.multi_hand_landmarks:
        pts = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
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
                mos_impact=-0.18,
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
