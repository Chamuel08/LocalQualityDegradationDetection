from __future__ import annotations

import uuid

import cv2
import numpy as np

from lqdd.config.loader import EdgeBleedConfig
from lqdd.detectors.base import bbox_from_mask, clip_bbox
from lqdd.models.enums import RegionType, RootCauseCategory, Severity
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput
from lqdd.models.report import DegradationItem, Evidence, RootCauseHypothesis


class EdgeBleedDetector:
    name = "edge_bleed"

    def __init__(self, config: EdgeBleedConfig) -> None:
        self.config = config

    def detect(
        self,
        frame_input: SingleFrameInput,
        scan_output: GlobalScanOutput,
    ) -> list[DegradationItem]:
        frame = frame_input.frame
        h, w = frame.shape[:2]
        edge_mask = scan_output.edge_mask
        if edge_mask is None or not edge_mask.any():
            edge_mask = np.zeros((h, w), dtype=bool)
            for nom in scan_output.nominations:
                if nom.region_type == int(RegionType.EDGE):
                    edge_mask = nom.mask
                    break

        if not edge_mask.any():
            return []

        b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        green_excess = g.astype(np.float32) - 0.5 * (r.astype(np.float32) + b.astype(np.float32))
        spill_pixels = (green_excess > self.config.green_channel_threshold * 255) & edge_mask
        spill_ratio = float(spill_pixels.sum()) / max(1, edge_mask.sum())

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_mask = ~scan_output.foreground_mask if scan_output.foreground_mask is not None else ~edge_mask
        if not bg_mask.any():
            bg_mask = ~edge_mask
        bg_mean = lab[bg_mask].mean(axis=0)
        edge_lab = lab[edge_mask]
        delta_e = np.sqrt(((edge_lab - bg_mean) ** 2).sum(axis=1)).mean() if edge_lab.size else 0.0

        severity, mos_impact = self._classify(spill_ratio, float(delta_e))
        if severity == Severity.GOOD:
            return []

        bbox = clip_bbox(bbox_from_mask(edge_mask), w, h)
        detail = (
            f"边缘带绿色溢出比例 {spill_ratio:.1%}，Lab ΔE 均值 {delta_e:.1f}；"
            f"超过阈值 spill≥{self.config.green_spill_minor:.0%} / ΔE≥{self.config.delta_e_spill_threshold}"
        )
        return [
            DegradationItem(
                degradation_id=f"deg_{uuid.uuid4().hex[:8]}",
                region_type=RegionType.EDGE.name.lower(),
                degradation_type="green_spill",
                severity=severity.value,
                confidence=min(0.95, 0.55 + spill_ratio),
                mos_impact=mos_impact,
                bbox=list(bbox),
                frame_indices=[scan_output.frame_index],
                description="人物轮廓边缘出现绿色溢色/抠像绿边",
                detector=self.name,
                evidence=Evidence(
                    method="green_spill_lab_delta_e",
                    metric="spill_ratio",
                    value=round(spill_ratio, 4),
                    threshold=self.config.green_spill_minor,
                    detail=detail,
                ),
                root_cause_hypothesis=RootCauseHypothesis(
                    cause=RootCauseCategory.MATTING_ERROR.value,
                    confidence=min(0.9, 0.5 + spill_ratio),
                ),
            )
        ]

    def _classify(self, spill_ratio: float, delta_e: float) -> tuple[Severity, float]:
        cfg = self.config
        if spill_ratio < cfg.green_spill_minor and delta_e < cfg.delta_e_spill_threshold:
            return Severity.GOOD, 0.0
        if spill_ratio >= cfg.green_spill_critical or delta_e >= cfg.delta_e_spill_threshold * 2:
            return Severity.CRITICAL, -0.8
        if spill_ratio >= cfg.green_spill_moderate:
            return Severity.MODERATE, -0.3
        return Severity.MINOR, -0.2
