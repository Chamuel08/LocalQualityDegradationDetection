"""Apply controlled degradations to portrait frames."""

from __future__ import annotations

import cv2
import numpy as np

from lqdd.detectors.base import (
    background_mask_from_scan,
    bbox_from_mask,
    compute_banding_score,
    face_region_mask,
    foreground_mask_from_scan,
    hair_region_mask,
)
from lqdd.models.inputs import GlobalScanOutput


def apply_green_edge(img: np.ndarray, edge_mask: np.ndarray, strength: float = 0.82) -> np.ndarray:
    out = img.copy()
    active = edge_mask if edge_mask is not None else np.zeros(img.shape[:2], dtype=bool)
    if not active.any():
        return out
    spill_bgr = np.array([20, 255, 20], dtype=np.float32)
    for c in range(3):
        ch = out[:, :, c].astype(np.float32)
        ch[active] = ch[active] * (1.0 - strength) + spill_bgr[c] * strength
        out[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    return out


def apply_blockiness(img: np.ndarray, scale: int = 8) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(4, scale)
    small = cv2.resize(img, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    noise = np.random.default_rng(0).integers(-6, 7, out.shape, dtype=np.int16)
    return np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_body_blur(img: np.ndarray, scan: GlobalScanOutput, sigma: float = 18.0) -> np.ndarray:
    """Blur the foreground bounding box (matches blur_artifact ROI crop semantics)."""
    h, w = img.shape[:2]
    fg = foreground_mask_from_scan(scan, h, w)
    out = img.copy()
    if not fg.any():
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    x, y, bw, bh = bbox_from_mask(fg)
    if bw <= 0 or bh <= 0:
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    patch = out[y : y + bh, x : x + bw]
    out[y : y + bh, x : x + bw] = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma)
    return out


def apply_mosaic(img: np.ndarray, scan: GlobalScanOutput, scale: int = 8) -> np.ndarray:
    """Pixelate portrait center band + foreground for mosaic_artifact global score."""
    h, w = img.shape[:2]
    fg = foreground_mask_from_scan(scan, h, w)
    small = cv2.resize(img, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = img.copy()
    target = fg.copy() if fg.any() else np.zeros((h, w), dtype=bool)
    target[int(h * 0.05) : int(h * 0.95), :] = True
    out[target] = mosaic[target]
    return out


def apply_banding(img: np.ndarray, scan: GlobalScanOutput, step: int = 10) -> np.ndarray:
    h, w = img.shape[:2]
    bg = background_mask_from_scan(scan, h, w)
    out = img.copy()
    if not bg.any():
        bg = np.ones((h, w), dtype=bool)
    band = out.copy()
    for y in range(h):
        val = int(band[y, 0].mean() // step) * step
        band[y, :] = (val, val // 2 + 10, 255 - val // 3)
    out[bg] = band[bg]
    return out


def apply_face_overexposure(img: np.ndarray, scan: GlobalScanOutput, gain: float = 1.35) -> np.ndarray:
    h, w = img.shape[:2]
    fg = foreground_mask_from_scan(scan, h, w)
    face = face_region_mask(fg, h, w)
    if not face.any():
        return img
    out = img.copy()
    patch = out[face].astype(np.float32) * gain + 35.0
    out[face] = np.clip(patch, 0, 255).astype(np.uint8)
    return out


def apply_hair_blur(img: np.ndarray, scan: GlobalScanOutput) -> np.ndarray:
    h, w = img.shape[:2]
    fg = foreground_mask_from_scan(scan, h, w)
    hair = hair_region_mask(fg, h, w)
    out = img.copy()
    if hair.sum() >= 40:
        ys, xs = np.where(hair)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        mean_color = out[hair].reshape(-1, 3).mean(axis=0)
        patch = np.full((y1 - y0, x1 - x0, 3), mean_color, dtype=np.uint8)
        patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=22.0)
        out[y0:y1, x0:x1] = patch
        out[hair] = mean_color.astype(np.uint8)
        return out
    y_end = max(24, int(h * 0.32))
    mean_color = out[:y_end].reshape(-1, 3).mean(axis=0)
    flat = np.full((y_end, w, 3), mean_color, dtype=np.uint8)
    out[:y_end, :] = cv2.GaussianBlur(flat, (0, 0), sigmaX=30.0)
    return out


def apply_degradation(
    category: str,
    img: np.ndarray,
    scan: GlobalScanOutput,
) -> np.ndarray:
    if category == "normal":
        return img.copy()
    if category == "edge":
        edge = scan.edge_mask
        if edge is None or not edge.any():
            edge = np.zeros(img.shape[:2], dtype=bool)
            edge[int(img.shape[0] * 0.15) : int(img.shape[0] * 0.85), int(img.shape[1] * 0.2) : int(img.shape[1] * 0.8)] = True
        return apply_green_edge(img, edge)
    if category == "block":
        return apply_blockiness(img)
    if category == "blur":
        return apply_body_blur(img, scan)
    if category == "mosaic":
        return apply_mosaic(img, scan)
    if category == "banding":
        return apply_banding(img, scan)
    if category == "face_over":
        return apply_face_overexposure(img, scan)
    if category == "hair_blur":
        return apply_hair_blur(img, scan)
    raise ValueError(f"unknown category: {category}")
