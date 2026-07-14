from __future__ import annotations

import cv2
import numpy as np

from lqdd.config.loader import HairTextureConfig
from lqdd.detectors.base import (
    bbox_from_mask,
    clip_bbox,
    face_region_mask,
    foreground_mask_from_scan,
    hair_region_mask,
    mask_from_heatmap,
)
from lqdd.detectors.helpers import make_degradation_item
from lqdd.models.enums import RegionType, RootCauseCategory, Severity
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput


def _laplacian_var(gray: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    mask_crop = mask[y0:y1, x0:x1]
    if crop.size < 9:
        return float(crop.var())
    fill = float(crop[mask_crop].mean()) if mask_crop.any() else float(crop.mean())
    crop[~mask_crop] = fill
    return float(cv2.Laplacian(crop.astype(np.uint8), cv2.CV_64F).var())


class HairTextureDetector:
    name = "hair_texture"

    def __init__(self, config: HairTextureConfig) -> None:
        self.config = config

    def detect(
        self,
        frame_input: SingleFrameInput,
        scan_output: GlobalScanOutput,
    ) -> list:
        frame = frame_input.frame
        h, w = frame.shape[:2]
        fg = foreground_mask_from_scan(scan_output, h, w)
        hair = hair_region_mask(fg, h, w)
        if hair.sum() < self.config.min_hair_pixels:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hair_var = _laplacian_var(gray, hair)
        face = face_region_mask(fg, h, w)
        face_var = _laplacian_var(gray, face) if face.sum() >= 120 else max(hair_var, 1.0)
        relative = hair_var / (face_var + 1e-6)

        absolute_hit = hair_var <= self.config.max_hair_laplacian_var
        relative_hit = face_var >= 10.0 and relative < self.config.relative_laplacian_threshold
        if not absolute_hit and not relative_hit:
            return []

        loss = float(
            np.clip(
                max(
                    1.0 - relative / max(self.config.relative_laplacian_threshold, 1e-3),
                    1.0 - hair_var / max(self.config.max_hair_laplacian_var, 1e-3),
                ),
                0,
                1,
            )
        )
        metric = "hair_face_laplacian_ratio" if relative_hit else "hair_laplacian_var"
        value = relative if relative_hit else hair_var
        threshold = (
            self.config.relative_laplacian_threshold
            if relative_hit
            else self.config.max_hair_laplacian_var
        )
        ys, xs = np.where(hair)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        heat = np.zeros((h, w), dtype=np.float32)
        heat[y0:y1, x0:x1] = loss
        region_mask = mask_from_heatmap(
            heat,
            threshold=0.25,
            roi_mask=hair,
            min_area=max(80, self.config.min_hair_pixels // 4),
        )
        bbox = clip_bbox(bbox_from_mask(region_mask) if region_mask is not None else (x0, y0, x1 - x0, y1 - y0), w, h)

        return [
            make_degradation_item(
                detector=self.name,
                degradation_type="hair_texture_loss",
                region_type=RegionType.HAIR,
                severity=Severity.MINOR.value,
                confidence=min(0.84, 0.58 + loss * 0.35),
                mos_impact=-0.2,
                bbox=bbox,
                region_mask=region_mask,
                method="laplacian_hair_vs_face",
                metric=metric,
                value=round(value, 4),
                threshold=threshold,
                detail=(
                    f"发丝 Laplacian 方差 {hair_var:.1f}，相对面部 {relative:.2f}；"
                    f"阈值 {self.config.max_hair_laplacian_var} / {self.config.relative_laplacian_threshold}"
                ),
                description="头发区域高频纹理不足或糊化",
                cause=RootCauseCategory.GENERATION_ARTIFACT,
                frame_index=scan_output.frame_index,
            )
        ]
