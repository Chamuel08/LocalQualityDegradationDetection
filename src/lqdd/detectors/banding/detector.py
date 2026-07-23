from __future__ import annotations

import cv2
import numpy as np

from lqdd.config.loader import BandingConfig
from lqdd.detectors.base import (
    background_mask_from_scan,
    bbox_from_mask,
    clip_bbox,
    compute_banding_score,
    localize_banding_mask,
)
from lqdd.detectors.helpers import make_degradation_item
from lqdd.models.enums import RegionType, RootCauseCategory, Severity
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput


class BandingArtifactDetector:
    name = "banding_artifact"

    def __init__(self, config: BandingConfig) -> None:
        self.config = config

    def detect(
        self,
        frame_input: SingleFrameInput,
        scan_output: GlobalScanOutput,
    ) -> list:
        frame = frame_input.frame
        h, w = frame.shape[:2]
        bg = background_mask_from_scan(scan_output, h, w)
        if bg.sum() < h * w * 0.08:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, bw, bh = bbox_from_mask(bg)
        crop = gray[y : y + bh, x : x + bw]
        if crop.size < 64:
            return []

        score = compute_banding_score(crop)
        if score < self.config.score_threshold:
            return []

        region_mask = localize_banding_mask(gray, self.config.localize_threshold, roi_mask=bg)
        bbox = clip_bbox(bbox_from_mask(region_mask) if region_mask is not None else (x, y, bw, bh), w, h)

        return [
            make_degradation_item(
                detector=self.name,
                degradation_type="banding",
                region_type=RegionType.BACKGROUND,
                severity=Severity.MINOR.value,
                confidence=min(0.82, 0.55 + (score - self.config.score_threshold) * 0.6),
                bbox=bbox,
                region_mask=region_mask,
                method="gradient_quantization_banding",
                metric="banding_score",
                value=score,
                threshold=self.config.score_threshold,
                detail=f"背景区域出现色带/量化台阶，banding_score={score:.2f} ≥ {self.config.score_threshold}",
                description="背景平滑区域出现色带伪影",
                cause=RootCauseCategory.ENCODING_LOSS,
                frame_index=scan_output.frame_index,
            )
        ]
