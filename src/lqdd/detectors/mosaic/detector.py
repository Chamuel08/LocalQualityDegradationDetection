from __future__ import annotations

import cv2
import numpy as np

from lqdd.config.loader import MosaicConfig
from lqdd.detectors.base import (
    bbox_from_mask,
    clip_bbox,
    compute_mosaic_score,
    detection_roi_mask,
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
