from __future__ import annotations

import cv2
import numpy as np

from lqdd.config.loader import GlobalScanConfig


def _fallback_segment_foreground(frame_bgr: np.ndarray) -> np.ndarray:
    """GrabCut-based fallback when MediaPipe is unavailable."""
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    rect = (int(w * 0.15), int(h * 0.08), int(w * 0.7), int(h * 0.84))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(frame_bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


def segment_foreground(frame_bgr: np.ndarray) -> np.ndarray:
    """MediaPipe Selfie Segmentation with GrabCut fallback."""
    try:
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = seg.process(rgb)
            if result.segmentation_mask is None:
                return _fallback_segment_foreground(frame_bgr)
            return result.segmentation_mask > 0.5
    except Exception:
        return _fallback_segment_foreground(frame_bgr)


def build_edge_band(foreground: np.ndarray, expand_px: int) -> np.ndarray:
    fg_u8 = (foreground.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1))
    dilated = cv2.dilate(fg_u8, kernel)
    eroded = cv2.erode(fg_u8, kernel)
    edge = (dilated > 0) & (eroded == 0)
    return edge


def build_segmentation_map(
    foreground: np.ndarray,
    edge: np.ndarray,
    overlay: np.ndarray,
    cfg: GlobalScanConfig,
) -> np.ndarray:
    from lqdd.models.enums import RegionType

    h, w = foreground.shape
    seg = np.full((h, w), RegionType.BACKGROUND, dtype=np.uint8)
    seg[foreground] = RegionType.BODY
    seg[edge] = RegionType.EDGE
    seg[overlay] = RegionType.TEXT_UI
    return seg
