from __future__ import annotations

import cv2
import numpy as np

from lqdd.config.loader import FaceArtifactConfig
from lqdd.detectors.base import (
    bbox_from_mask,
    clip_bbox,
    face_region_mask,
    foreground_mask_from_scan,
    mask_from_heatmap,
)
from lqdd.detectors.helpers import make_degradation_item
from lqdd.models.enums import RegionType, RootCauseCategory, Severity
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput


class FaceArtifactDetector:
    """Face ROI quality check (Laplacian + exposure).

    Full ArcFace embedding drift is optional via ``mediapipe``/InsightFace extras;
    this default path stays lightweight for the GitHub demo.
    """

    name = "face_artifact"

    def __init__(self, config: FaceArtifactConfig) -> None:
        self.config = config

    def detect(
        self,
        frame_input: SingleFrameInput,
        scan_output: GlobalScanOutput,
    ) -> list:
        frame = frame_input.frame
        h, w = frame.shape[:2]
        fg = foreground_mask_from_scan(scan_output, h, w)
        face = face_region_mask(fg, h, w)
        if face.sum() < self.config.min_face_pixels:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        face_l = l_channel[face]
        over_ratio = float((face_l > 220).mean())
        lap_var = float(cv2.Laplacian(gray[face].reshape(-1, 1), cv2.CV_64F).var()) if face.any() else 0.0
        ys, xs = np.where(face)
        crop = gray[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
        if crop.size >= 16:
            lap_var = float(cv2.Laplacian(crop, cv2.CV_64F).var())

        over_hit = over_ratio >= self.config.overexposure_ratio_threshold
        blur_hit = lap_var <= self.config.blur_laplacian_threshold
        if not over_hit and not blur_hit:
            return []

        if over_hit and blur_hit:
            metric = "face_composite"
            value = max(over_ratio, 1.0 - lap_var / self.config.blur_laplacian_threshold)
            threshold = 0.5
            detail = f"面部过曝比例 {over_ratio:.1%}，Laplacian var={lap_var:.0f}"
            degradation_type = "face_artifact"
        elif over_hit:
            metric = "face_overexposure_ratio"
            value = over_ratio
            threshold = self.config.overexposure_ratio_threshold
            detail = f"面部过曝像素占比 {over_ratio:.1%} ≥ {threshold:.0%}"
            degradation_type = "overexposure"
        else:
            metric = "face_laplacian_var"
            value = lap_var
            threshold = self.config.blur_laplacian_threshold
            detail = f"面部 Laplacian 方差 {lap_var:.0f} ≤ {threshold:.0f}（偏糊）"
            degradation_type = "face_blur"

        heat = np.zeros((h, w), dtype=np.float32)
        heat[face] = 1.0 if over_hit else float(np.clip(1.0 - lap_var / threshold, 0, 1))
        region_mask = mask_from_heatmap(heat, threshold=0.35, roi_mask=face, min_area=150)
        bbox = clip_bbox(bbox_from_mask(region_mask) if region_mask is not None else bbox_from_mask(face), w, h)

        return [
            make_degradation_item(
                detector=self.name,
                degradation_type=degradation_type,
                region_type=RegionType.FACE,
                severity=Severity.MINOR.value,
                confidence=0.68 if blur_hit else 0.72,
                mos_impact=-0.22,
                bbox=bbox,
                region_mask=region_mask,
                method="face_roi_exposure_laplacian",
                metric=metric,
                value=value,
                threshold=threshold,
                detail=detail,
                description="面部区域出现过曝或细节损失",
                cause=RootCauseCategory.GENERATION_ARTIFACT,
                frame_index=scan_output.frame_index,
            )
        ]
