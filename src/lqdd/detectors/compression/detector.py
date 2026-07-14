from __future__ import annotations

import uuid
from dataclasses import dataclass

import cv2
import numpy as np

from lqdd.config.loader import CompressionConfig
from lqdd.detectors.base import (
    blockiness_on_mask,
    clip_bbox,
    compute_blockiness_score,
    compute_texture_loss_score,
    detection_roi_mask,
    is_ai_generated_style,
    localize_blockiness_bbox,
    localize_blockiness_mask,
    localize_texture_loss_bbox,
    localize_texture_loss_mask,
)
from lqdd.report.mask_codec import encode_mask_rle
from lqdd.models.enums import RegionType, RootCauseCategory, Severity
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput
from lqdd.models.report import DegradationItem, Evidence, RootCauseHypothesis


@dataclass
class CompressionSignals:
    blockiness_score: float
    texture_loss_score: float
    edge_blockiness_score: float | None
    edge_block_ratio: float | None
    method: str
    metric: str
    value: float
    threshold: float
    detail: str


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
        signals = self._analyze(gray, scan_output)
        severity, mos_impact = self._classify(signals)
        if severity == Severity.GOOD:
            return []

        overlay = scan_output.overlay_mask
        roi = detection_roi_mask(scan_output, h, w)
        if overlay is not None and overlay.any():
            roi &= ~overlay
        bbox = clip_bbox(self._localize_bbox(gray, signals, roi, scan_output), w, h)
        region_mask = self._localize_mask(gray, signals, roi, scan_output, w, h)
        mask_rle = encode_mask_rle(region_mask) if region_mask is not None and region_mask.any() else None
        return [
            DegradationItem(
                degradation_id=f"deg_{uuid.uuid4().hex[:8]}",
                region_type=RegionType.BACKGROUND.name.lower(),
                degradation_type="blockiness",
                severity=severity.value,
                confidence=self._confidence(signals, severity),
                mos_impact=mos_impact,
                bbox=list(bbox),
                region_mask_rle=mask_rle,
                frame_indices=[scan_output.frame_index],
                description="画面出现块状压缩伪影/高频纹理损失",
                detector=self.name,
                evidence=Evidence(
                    method=signals.method,
                    metric=signals.metric,
                    value=round(signals.value, 4),
                    threshold=signals.threshold,
                    detail=signals.detail,
                ),
                root_cause_hypothesis=RootCauseHypothesis(
                    cause=RootCauseCategory.ENCODING_LOSS.value,
                    confidence=min(0.88, 0.45 + max(0.0, signals.value - signals.threshold) * 0.3),
                ),
            )
        ]

    def _localize_bbox(
        self,
        gray: np.ndarray,
        signals: CompressionSignals,
        roi: np.ndarray,
        scan_output: GlobalScanOutput,
    ) -> tuple[int, int, int, int]:
        cfg = self.config
        tight: tuple[int, int, int, int] | None = None

        if signals.metric == "texture_loss_score":
            tight = localize_texture_loss_bbox(
                gray, cfg.texture_var_reference, cfg.texture_loss_threshold, roi_mask=roi
            )
        elif signals.metric == "edge_block_ratio":
            edge = scan_output.edge_mask
            tight = localize_blockiness_bbox(gray, 1.0, roi_mask=edge if edge is not None else roi)
        elif signals.metric == "blockiness_score":
            tile_thr = (
                cfg.mild_blockiness_threshold - 0.01
                if signals.method == "dct_blockiness_mild"
                else cfg.blockiness_coarse_threshold * 0.85
            )
            tight = localize_blockiness_bbox(gray, tile_thr, roi_mask=roi)

        if tight is not None and tight[2] > 0 and tight[3] > 0:
            return tight
        h, w = gray.shape
        peak_y, peak_x = np.unravel_index(int(np.argmax(gray)), gray.shape)
        half = max(64, min(h, w) // 10)
        x0 = max(0, peak_x - half)
        y0 = max(0, peak_y - half)
        return x0, y0, min(w - x0, half * 2), min(h - y0, half * 2)

    def _localize_mask(
        self,
        gray: np.ndarray,
        signals: CompressionSignals,
        roi: np.ndarray,
        scan_output: GlobalScanOutput,
        w: int,
        h: int,
    ) -> np.ndarray | None:
        cfg = self.config
        mask: np.ndarray | None = None

        if signals.metric == "texture_loss_score":
            mask = localize_texture_loss_mask(
                gray, cfg.texture_var_reference, cfg.texture_loss_threshold, roi_mask=roi
            )
        elif signals.metric == "edge_block_ratio":
            edge = scan_output.edge_mask
            mask = localize_blockiness_mask(gray, 1.0, roi_mask=edge if edge is not None else roi)
        elif signals.metric == "blockiness_score":
            tile_thr = (
                cfg.mild_blockiness_threshold - 0.01
                if signals.method == "dct_blockiness_mild"
                else cfg.blockiness_coarse_threshold * 0.85
            )
            mask = localize_blockiness_mask(gray, tile_thr, roi_mask=roi)

        if mask is not None and mask.any():
            return mask

        bbox = self._localize_bbox(gray, signals, roi, scan_output)
        x, y, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            return None
        fallback = np.zeros((h, w), dtype=bool)
        fallback[y : y + bh, x : x + bw] = True
        return fallback

    def _analyze(self, gray: np.ndarray, scan_output: GlobalScanOutput) -> CompressionSignals:
        cfg = self.config
        block = compute_blockiness_score(gray)
        texture = compute_texture_loss_score(gray, cfg.texture_var_reference)
        edge_block = blockiness_on_mask(gray, scan_output.edge_mask) if scan_output.edge_mask is not None else None
        edge_ratio = (edge_block / block) if edge_block is not None and block > 0 else None

        # AI 合成图（扩散模型等）天然具有极低 Laplacian 方差，导致 texture_loss_score
        # 接近 1.0，若不加以识别会产生大量误报（将 AI 柔和风格误判为压缩失真）。
        # 当检测到 AI 合成风格时，texture_loss 分支整体跳过，只保留 DCT blockiness 路径
        # （真实压缩 blockiness 不受 AI 风格影响）。
        ai_style = is_ai_generated_style(gray)

        if block >= cfg.blockiness_coarse_threshold:
            return CompressionSignals(
                blockiness_score=block,
                texture_loss_score=texture,
                edge_blockiness_score=edge_block,
                edge_block_ratio=edge_ratio,
                method="dct_blockiness",
                metric="blockiness_score",
                value=block,
                threshold=cfg.blockiness_threshold,
                detail=(
                    f"8×8 DCT 块边界能量比 {block:.2f}；"
                    f"超过阈值 {cfg.blockiness_threshold}"
                ),
            )
        if texture >= cfg.texture_loss_threshold and not ai_style:
            # ai_style=True 时跳过：AI 合成图低 Laplacian 方差是正常风格，
            # 不代表真实的高频信息丢失，不应被判为压缩伪影。
            return CompressionSignals(
                blockiness_score=block,
                texture_loss_score=texture,
                edge_blockiness_score=edge_block,
                edge_block_ratio=edge_ratio,
                method="laplacian_texture_loss",
                metric="texture_loss_score",
                value=texture,
                threshold=cfg.texture_loss_threshold,
                detail=(
                    f"全图高频纹理损失 {texture:.1%}（Laplacian 方差相对参考 {cfg.texture_var_reference:.0f}）；"
                    f"超过阈值 {cfg.texture_loss_threshold:.0%}"
                ),
            )
        if (
            edge_block is not None
            and edge_ratio is not None
            and edge_ratio >= cfg.edge_block_ratio_threshold
            and edge_block >= 1.0
        ):
            return CompressionSignals(
                blockiness_score=block,
                texture_loss_score=texture,
                edge_blockiness_score=edge_block,
                edge_block_ratio=edge_ratio,
                method="edge_blockiness_ratio",
                metric="edge_block_ratio",
                value=edge_ratio,
                threshold=cfg.edge_block_ratio_threshold,
                detail=(
                    f"轮廓边缘带块效应偏强：edge_block={edge_block:.2f}，"
                    f"相对全图比值 {edge_ratio:.3f} ≥ {cfg.edge_block_ratio_threshold}"
                ),
            )
        if block >= cfg.mild_blockiness_threshold:
            return CompressionSignals(
                blockiness_score=block,
                texture_loss_score=texture,
                edge_blockiness_score=edge_block,
                edge_block_ratio=edge_ratio,
                method="dct_blockiness_mild",
                metric="blockiness_score",
                value=block,
                threshold=cfg.mild_blockiness_threshold,
                detail=(
                    f"全图轻度块边界能量升高 {block:.2f}；"
                    f"超过 mild 阈值 {cfg.mild_blockiness_threshold}"
                ),
            )

        return CompressionSignals(
            blockiness_score=block,
            texture_loss_score=texture,
            edge_blockiness_score=edge_block,
            edge_block_ratio=edge_ratio,
            method="dct_blockiness",
            metric="blockiness_score",
            value=block,
            threshold=cfg.blockiness_coarse_threshold,
            detail=f"未超过压缩检测阈值（block={block:.2f}, texture={texture:.2f}）",
        )

    def _classify(self, signals: CompressionSignals) -> tuple[Severity, float]:
        cfg = self.config
        if signals.metric == "blockiness_score":
            score = signals.blockiness_score
            if score < cfg.mild_blockiness_threshold:
                return Severity.GOOD, 0.0
            if score < cfg.blockiness_coarse_threshold:
                return Severity.MINOR, -0.15
            if score >= cfg.blockiness_threshold * 1.3:
                return Severity.SEVERE, -0.4
            if score >= cfg.blockiness_threshold:
                return Severity.MODERATE, -0.3
            return Severity.MINOR, -0.2
        if signals.metric == "texture_loss_score":
            score = signals.texture_loss_score
            if score >= cfg.texture_loss_threshold * 1.4:
                return Severity.MODERATE, -0.35
            return Severity.MINOR, -0.25
        if signals.metric == "edge_block_ratio":
            return Severity.MINOR, -0.2
        return Severity.GOOD, 0.0

    def _confidence(self, signals: CompressionSignals, severity: Severity) -> float:
        base = {
            Severity.MINOR: 0.58,
            Severity.MODERATE: 0.72,
            Severity.SEVERE: 0.85,
        }.get(severity, 0.5)
        margin = max(0.0, signals.value - signals.threshold)
        return min(0.92, base + margin * 0.25)
