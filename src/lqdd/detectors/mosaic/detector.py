from __future__ import annotations

import cv2
import numpy as np

from lqdd.config.loader import MosaicConfig
from lqdd.detectors.base import (
    bbox_from_mask,
    clip_bbox,
    compute_mosaic_score,
    detection_roi_mask,
    is_ai_generated_style,
    localize_mosaic_mask,
)
from lqdd.detectors.helpers import make_degradation_item
from lqdd.models.enums import RegionType, RootCauseCategory, Severity
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput


class MosaicArtifactDetector:
    name = "mosaic_artifact"

    def __init__(self, config: MosaicConfig) -> None:
        self.config = config

    def detect(
        self,
        frame_input: SingleFrameInput,
        scan_output: GlobalScanOutput,
    ) -> list:
        frame = frame_input.frame
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = compute_mosaic_score(gray)
        if score < self.config.score_threshold:
            return []

        # compute_mosaic_score 的 flat_ratio 分量会在 AI 合成图的大面积平滑背景上虚高：
        # AI 图的背景色块平整，8×8 block 方差天然 < 12，导致 flat_ratio 接近 1，
        # 从而使总分虚假超过阈值，将"AI 风格平滑"误判为"马赛克"。
        # 当检测到 AI 合成风格时，直接跳过马赛克检测。
        if is_ai_generated_style(gray):
            return []

        roi = detection_roi_mask(scan_output, h, w)
        overlay = scan_output.overlay_mask
        if overlay is not None:
            roi &= ~overlay

        region_mask = localize_mosaic_mask(gray, self.config.localize_threshold, roi_mask=roi)
        bbox = clip_bbox(
            bbox_from_mask(region_mask) if region_mask is not None else (0, 0, w // 2, h // 2),
            w,
            h,
        )

        severity = Severity.MODERATE.value if score >= self.config.score_threshold * 1.2 else Severity.MINOR.value
        mos = -0.35 if severity == Severity.MODERATE.value else -0.25

        return [
            make_degradation_item(
                detector=self.name,
                degradation_type="mosaic",
                region_type=RegionType.BODY,
                severity=severity,
                confidence=min(0.88, 0.58 + (score - self.config.score_threshold) * 0.5),
                mos_impact=mos,
                bbox=bbox,
                region_mask=region_mask,
                method="block_flatness_mosaic",
                metric="mosaic_score",
                value=score,
                threshold=self.config.score_threshold,
                detail=f"检测到马赛克/像素块平铺特征，score={score:.2f} ≥ {self.config.score_threshold}",
                description="画面出现马赛克或过度像素化块效应",
                cause=RootCauseCategory.ENCODING_LOSS,
                frame_index=scan_output.frame_index,
            )
        ]
