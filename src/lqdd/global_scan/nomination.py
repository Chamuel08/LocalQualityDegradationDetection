from __future__ import annotations

import numpy as np

from lqdd.config.loader import GlobalScanConfig
from lqdd.detectors.base import bbox_from_mask, compute_blockiness_score
from lqdd.models.enums import RegionType
from lqdd.models.inputs import RegionNomination


def build_nominations(
    frame_bgr: np.ndarray,
    foreground: np.ndarray,
    edge_mask: np.ndarray,
    overlay_mask: np.ndarray,
    cfg: GlobalScanConfig,
) -> list[RegionNomination]:
    nominations: list[RegionNomination] = []
    h, w = frame_bgr.shape[:2]
    fg_area = foreground.sum()
    if fg_area < h * w * 0.01:
        return nominations

    edge_active = edge_mask & ~overlay_mask
    if edge_active.any():
        bbox = bbox_from_mask(edge_active)
        b, g, r = frame_bgr[:, :, 0], frame_bgr[:, :, 1], frame_bgr[:, :, 2]
        green_excess = g.astype(np.float32) - 0.5 * (r.astype(np.float32) + b.astype(np.float32))
        edge_vals = green_excess[edge_active]
        green_bias = float(np.clip(edge_vals.mean() / 128.0, 0, 1)) if edge_vals.size else 0.0
        nominations.append(
            RegionNomination(
                region_type=int(RegionType.EDGE),
                bbox=bbox,
                mask=edge_active,
                anomaly_score=min(1.0, 0.4 + green_bias),
                confidence=min(1.0, 0.5 + green_bias * 0.4),
                suggested_detectors=["edge_bleed"],
                features={"green_channel_bias": green_bias},
            )
        )

    gray = np.dot(frame_bgr[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
    blockiness = compute_blockiness_score(gray)
    block_hint = float(np.clip((blockiness - 1.0) / 2.0, 0, 1))
    if blockiness >= cfg.nomination_threshold or block_hint > 0.3:
        bg_mask = (~foreground) & (~overlay_mask)
        if not bg_mask.any():
            bg_mask = np.ones((h, w), dtype=bool)
        nominations.append(
            RegionNomination(
                region_type=int(RegionType.BACKGROUND),
                bbox=bbox_from_mask(bg_mask),
                mask=bg_mask,
                anomaly_score=block_hint,
                confidence=min(1.0, 0.45 + block_hint * 0.4),
                suggested_detectors=["compression_artifact"],
                features={"blockiness_hint": block_hint, "blockiness_score": blockiness},
            )
        )
    return nominations
