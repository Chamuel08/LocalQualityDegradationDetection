from __future__ import annotations

import cv2
import numpy as np

from lqdd.config.loader import GlobalScanConfig
from lqdd.models.inputs import BBox


def detect_text_ui_bands(frame_bgr: np.ndarray, cfg: GlobalScanConfig) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    band_h = int(h * cfg.subtitle_band_ratio)
    if band_h > 0:
        bottom = frame_bgr[h - band_h : h, :]
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        density = edges.mean() / 255.0
        if density >= cfg.text_edge_density:
            mask[h - band_h : h, :] = True
    return mask


def merge_ignore_regions(
    text_ui_mask: np.ndarray,
    ignore_regions: list[BBox] | None,
    h: int,
    w: int,
) -> np.ndarray:
    overlay = text_ui_mask.copy()
    if not ignore_regions:
        return overlay
    for x, y, bw, bh in ignore_regions:
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        x1 = max(0, x)
        y1 = max(0, y)
        overlay[y1:y2, x1:x2] = True
    return overlay
