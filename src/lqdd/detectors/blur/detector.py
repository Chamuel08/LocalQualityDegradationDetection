from __future__ import annotations

import cv2
import numpy as np

from lqdd.config.loader import BlurConfig
from lqdd.detectors.base import (
    bbox_from_mask,
    clip_bbox,
    compute_texture_loss_score,
    foreground_mask_from_scan,
    is_ai_generated_style,
    localize_texture_loss_mask,
)
from lqdd.detectors.helpers import make_degradation_item
from lqdd.models.enums import RegionType, RootCauseCategory, Severity
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput


class BlurArtifactDetector:
    name = "blur_artifact"

    def __init__(self, config: BlurConfig) -> None:
        self.config = config

    def detect(
        self,
        frame_input: SingleFrameInput,
        scan_output: GlobalScanOutput,
    ) -> list:
        frame = frame_input.frame
        h, w = frame.shape[:2]
        fg = foreground_mask_from_scan(scan_output, h, w)
        if fg.sum() < h * w * self.config.min_foreground_ratio:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, bw, bh = bbox_from_mask(fg)
        crop = gray[y : y + bh, x : x + bw]
        if crop.size < 64:
            return []

        texture_loss = compute_texture_loss_score(crop, self.config.texture_var_reference)
        lap_var = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        if texture_loss < self.config.texture_loss_threshold:
            return []

        # AI 合成图（扩散模型）天然具有"柔和"低纹理风格，
        # compute_texture_loss_score 基于 reference_var=2400 会把它误判为模糊。
        # 用 is_ai_generated_style() 识别此类图像并直接跳过，
        # 避免将 AI 风格柔化报告为 blur_artifact（误判根因也错误，encoding_loss）。
        if is_ai_generated_style(gray):
            return []

        roi = fg.copy()
        overlay = scan_output.overlay_mask
        if overlay is not None:
            roi &= ~overlay

        region_mask = localize_texture_loss_mask(
            gray,
            self.config.texture_var_reference,
            self.config.texture_loss_threshold,
            roi_mask=roi,
        )
        bbox = clip_bbox(bbox_from_mask(region_mask) if region_mask is not None else (x, y, bw, bh), w, h)

        if texture_loss >= self.config.texture_loss_threshold * 1.35:
            severity = Severity.MODERATE.value
            mos = -0.32
            conf = 0.74
        else:
            severity = Severity.MINOR.value
            mos = -0.22
            conf = 0.62

        return [
            make_degradation_item(
                detector=self.name,
                degradation_type="blur",
                region_type=RegionType.BODY,
                severity=severity,
                confidence=conf,
                mos_impact=mos,
                bbox=bbox,
                region_mask=region_mask,
                method="laplacian_regional_blur",
                metric="texture_loss_score",
                value=texture_loss,
                threshold=self.config.texture_loss_threshold,
                detail=(
                    f"主体区域高频纹理损失 {texture_loss:.1%}（Laplacian var≈{lap_var:.0f}）；"
                    f"超过阈值 {self.config.texture_loss_threshold:.0%}"
                ),
                description="主体区域出现区域性模糊/纹理损失",
                cause=RootCauseCategory.ENCODING_LOSS,
                frame_index=scan_output.frame_index,
            )
        ]
