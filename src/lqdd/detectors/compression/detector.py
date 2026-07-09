from __future__ import annotations

import uuid

import cv2
import numpy as np

from lqdd.config.loader import CompressionConfig
from lqdd.detectors.base import bbox_from_mask, clip_bbox, compute_blockiness_score
from lqdd.models.enums import RegionType, RootCauseCategory, Severity
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput
from lqdd.models.report import DegradationItem, Evidence, RootCauseHypothesis


class CompressionArtifactDetector:
    name = "compression_artifact"

    def __init__(self, config: CompressionConfig) -> None:
        self.config = config

    def detect(
        self,
        frame_input: SingleFrameInput,
        scan_output: GlobalScanOutput,
    ) -> list[DegradationItem]:
        frame = frame_input.frame
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = compute_blockiness_score(gray)
        severity, mos_impact = self._classify(score)
        if severity == Severity.GOOD:
            return []

        overlay = scan_output.overlay_mask
        if overlay is not None and overlay.any():
            mask = ~overlay
        else:
            mask = np.ones((h, w), dtype=bool)
        bbox = clip_bbox(bbox_from_mask(mask), w, h)
        detail = (
            f"8×8 DCT 块边界能量比 {score:.2f}；"
            f"超过阈值 {self.config.blockiness_threshold}"
        )
        return [
            DegradationItem(
                degradation_id=f"deg_{uuid.uuid4().hex[:8]}",
                region_type=RegionType.BACKGROUND.name.lower(),
                degradation_type="blockiness",
                severity=severity.value,
                confidence=min(0.92, 0.5 + (score - 1.0) * 0.2),
                mos_impact=mos_impact,
                bbox=list(bbox),
                frame_indices=[scan_output.frame_index],
                description="画面出现块状压缩伪影/马赛克感",
                detector=self.name,
                evidence=Evidence(
                    method="dct_blockiness",
                    metric="blockiness_score",
                    value=round(score, 4),
                    threshold=self.config.blockiness_threshold,
                    detail=detail,
                ),
                root_cause_hypothesis=RootCauseHypothesis(
                    cause=RootCauseCategory.ENCODING_LOSS.value,
                    confidence=min(0.88, 0.45 + (score - 1.0) * 0.15),
                ),
            )
        ]

    def _classify(self, score: float) -> tuple[Severity, float]:
        if score < self.config.blockiness_coarse_threshold:
            return Severity.GOOD, 0.0
        if score >= self.config.blockiness_threshold * 1.3:
            return Severity.SEVERE, -0.4
        if score >= self.config.blockiness_threshold:
            return Severity.MODERATE, -0.3
        return Severity.MINOR, -0.2
