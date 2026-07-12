from __future__ import annotations

import cv2
import numpy as np

from lqdd.config.loader import BackgroundArtifactConfig
from lqdd.detectors.base import (
    background_mask_from_scan,
    bbox_from_mask,
    blockiness_on_mask,
    clip_bbox,
    localize_blockiness_mask,
)
from lqdd.detectors.helpers import make_degradation_item
from lqdd.models.enums import RegionType, RootCauseCategory, Severity
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput


class BackgroundArtifactDetector:
    name = "background_artifact"

    def __init__(self, config: BackgroundArtifactConfig) -> None:
        self.config = config

    def detect(
        self,
        frame_input: SingleFrameInput,
        scan_output: GlobalScanOutput,
    ) -> list:
        frame = frame_input.frame
        h, w = frame.shape[:2]
        bg = background_mask_from_scan(scan_output, h, w)
        if bg.sum() < h * w * self.config.min_background_ratio:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        block = blockiness_on_mask(gray, bg)
        if block is None:
            return []

        bgr_bg = frame[bg]
        channel_std = float(np.std(bgr_bg.astype(np.float32), axis=0).mean())
        color_cast = float(np.clip(1.0 - channel_std / 64.0, 0, 1))

        triggered = block >= self.config.blockiness_threshold
        if not triggered and color_cast >= self.config.color_cast_threshold:
            triggered = block >= 1.08 and color_cast >= self.config.color_cast_threshold
        if not triggered:
            return []

        metric = "background_blockiness" if block >= self.config.blockiness_threshold else "background_color_cast"
        value = block if metric == "background_blockiness" else color_cast
        threshold = (
            self.config.blockiness_threshold
            if metric == "background_blockiness"
            else self.config.color_cast_threshold
        )

        region_mask = localize_blockiness_mask(
            gray,
            self.config.blockiness_threshold * 0.85,
            roi_mask=bg,
        )
        if region_mask is None or not region_mask.any():
            region_mask = bg
        bbox = clip_bbox(bbox_from_mask(region_mask), w, h)

        return [
            make_degradation_item(
                detector=self.name,
                degradation_type="background_artifact",
                region_type=RegionType.BACKGROUND,
                severity=Severity.MINOR.value,
                confidence=min(0.85, 0.56 + max(0.0, value - threshold) * 0.4),
                mos_impact=-0.18,
                bbox=bbox,
                region_mask=region_mask,
                method="background_blockiness_color_cast",
                metric=metric,
                value=value,
                threshold=threshold,
                detail=(
                    f"背景区域 blockiness={block:.2f}，color_cast={color_cast:.2f}；"
                    f"独立于全图 compression 的背景专项检测"
                ),
                description="背景区域出现块效应或色彩漂移",
                cause=RootCauseCategory.ENCODING_LOSS,
                frame_index=scan_output.frame_index,
            )
        ]
