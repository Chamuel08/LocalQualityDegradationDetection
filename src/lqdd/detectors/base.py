from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput
from lqdd.models.report import DegradationItem


class Detector(Protocol):
    name: str

    def detect(
        self,
        frame_input: SingleFrameInput,
        scan_output: GlobalScanOutput,
    ) -> list[DegradationItem]:
        ...


def clip_bbox(bbox: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x, y, bw, bh = bbox
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))
    return x, y, bw, bh


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _tile_heatmap(
    gray: np.ndarray,
    scorer,
    tile_size: int = 96,
    stride: int = 48,
) -> np.ndarray:
    h, w = gray.shape
    heat = np.zeros((h, w), dtype=np.float32)
    for y0 in range(0, max(1, h - tile_size + 1), stride):
        for x0 in range(0, max(1, w - tile_size + 1), stride):
            y1 = min(h, y0 + tile_size)
            x1 = min(w, x0 + tile_size)
            patch = gray[y0:y1, x0:x1]
            if patch.shape[0] < 24 or patch.shape[1] < 24:
                continue
            score = float(scorer(patch))
            heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], score)
    return heat


def _morph_component_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    return mask_u8, labels, stats


def _best_component_mask(
    mask_u8: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    min_area: int,
) -> tuple[tuple[int, int, int, int] | None, np.ndarray | None]:
    best_i = -1
    best_area = 0
    for i in range(1, stats.shape[0]):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_area = area
            best_i = i
    if best_i < 0:
        return None, None
    comp = labels == best_i
    x = int(stats[best_i, cv2.CC_STAT_LEFT])
    y = int(stats[best_i, cv2.CC_STAT_TOP])
    bw = int(stats[best_i, cv2.CC_STAT_WIDTH])
    bh = int(stats[best_i, cv2.CC_STAT_HEIGHT])
    return (x, y, bw, bh), comp


def _heatmap_thresholds(
    heat: np.ndarray,
    threshold: float,
    roi_mask: np.ndarray | None,
) -> list[float]:
    thresholds = [threshold]
    if roi_mask is not None and roi_mask.any():
        roi_vals = heat[roi_mask]
    else:
        roi_vals = heat.ravel()
    roi_vals = roi_vals[roi_vals > 0]
    if len(roi_vals):
        for pct in (85, 88, 90, 92, 94, 96):
            thresholds.append(max(threshold, float(np.percentile(roi_vals, pct))))
    return sorted(set(thresholds))


def _pick_heatmap_component(
    heat: np.ndarray,
    threshold: float,
    roi_mask: np.ndarray | None = None,
    min_area: int = 400,
) -> tuple[tuple[int, int, int, int] | None, np.ndarray | None]:
    mask = heat >= threshold
    if roi_mask is not None:
        mask &= roi_mask
    if not mask.any():
        return None, None

    mask_u8, labels, stats = _morph_component_masks(mask)
    bbox, comp = _best_component_mask(mask_u8, labels, stats, min_area)
    if comp is not None:
        return bbox, comp

    peak = float(heat.max())
    if peak < threshold:
        return None, None
    peak_mask = heat >= (peak * 0.92)
    if roi_mask is not None:
        peak_mask &= roi_mask
    return bbox_from_mask(peak_mask) if peak_mask.any() else None, peak_mask if peak_mask.any() else None


def mask_from_heatmap(
    heat: np.ndarray,
    threshold: float,
    roi_mask: np.ndarray | None = None,
    min_area: int = 400,
    max_area_ratio: float = 0.15,
) -> np.ndarray | None:
    h, w = heat.shape
    max_area = int(h * w * max_area_ratio)
    fallback: np.ndarray | None = None
    fallback_area = h * w

    for thr in _heatmap_thresholds(heat, threshold, roi_mask):
        bbox, comp = _pick_heatmap_component(heat, thr, roi_mask, min_area)
        if comp is None:
            continue
        area = int(comp.sum())
        if area <= max_area:
            return comp
        if area < fallback_area:
            fallback = comp
            fallback_area = area

    if fallback is not None:
        return fallback

    peak_y, peak_x = np.unravel_index(int(heat.argmax()), heat.shape)
    half = max(48, min(h, w) // 8)
    out = np.zeros((h, w), dtype=bool)
    x0 = max(0, peak_x - half)
    y0 = max(0, peak_y - half)
    x1 = min(w, peak_x + half)
    y1 = min(h, peak_y + half)
    out[y0:y1, x0:x1] = True
    return out


def bbox_from_heatmap(
    heat: np.ndarray,
    threshold: float,
    roi_mask: np.ndarray | None = None,
    min_area: int = 400,
    max_area_ratio: float = 0.15,
) -> tuple[int, int, int, int] | None:
    h, w = heat.shape
    max_area = int(h * w * max_area_ratio)

    fallback: tuple[int, int, int, int] | None = None
    for thr in _heatmap_thresholds(heat, threshold, roi_mask):
        bbox, comp = _pick_heatmap_component(heat, thr, roi_mask, min_area)
        if bbox is None:
            continue
        area = bbox[2] * bbox[3]
        if area <= max_area:
            return bbox
        if fallback is None or area < fallback[2] * fallback[3]:
            fallback = bbox

    if fallback is not None:
        return fallback

    peak_y, peak_x = np.unravel_index(int(heat.argmax()), heat.shape)
    half = max(48, min(h, w) // 8)
    x0 = max(0, peak_x - half)
    y0 = max(0, peak_y - half)
    x1 = min(w, peak_x + half)
    y1 = min(h, peak_y + half)
    return x0, y0, x1 - x0, y1 - y0


def localize_texture_loss_bbox(
    gray: np.ndarray,
    reference_var: float,
    loss_threshold: float,
    roi_mask: np.ndarray | None = None,
) -> tuple[int, int, int, int] | None:
    mask = localize_texture_loss_mask(gray, reference_var, loss_threshold, roi_mask)
    return bbox_from_mask(mask) if mask is not None and mask.any() else None


def localize_texture_loss_mask(
    gray: np.ndarray,
    reference_var: float,
    loss_threshold: float,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    heat = _tile_heatmap(gray, lambda p: compute_texture_loss_score(p, reference_var))
    tile_thr = max(0.12, loss_threshold * 0.65)
    return mask_from_heatmap(heat, tile_thr, roi_mask=roi_mask, min_area=300)


def localize_blockiness_bbox(
    gray: np.ndarray,
    score_threshold: float,
    roi_mask: np.ndarray | None = None,
) -> tuple[int, int, int, int] | None:
    mask = localize_blockiness_mask(gray, score_threshold, roi_mask)
    return bbox_from_mask(mask) if mask is not None and mask.any() else None


def localize_blockiness_mask(
    gray: np.ndarray,
    score_threshold: float,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    heat = _tile_heatmap(gray, compute_blockiness_score, tile_size=80, stride=40)
    return mask_from_heatmap(heat, score_threshold, roi_mask=roi_mask, min_area=250)


def localize_spill_bbox(spill_mask: np.ndarray, min_area: int = 50) -> tuple[int, int, int, int] | None:
    mask = localize_spill_mask(spill_mask, min_area)
    return bbox_from_mask(mask) if mask is not None and mask.any() else None


def localize_spill_mask(spill_mask: np.ndarray, min_area: int = 50) -> np.ndarray | None:
    if not spill_mask.any():
        return None
    return mask_from_heatmap(
        spill_mask.astype(np.float32),
        threshold=0.5,
        roi_mask=None,
        min_area=min_area,
    )


def detection_roi_mask(scan_output: GlobalScanOutput, h: int, w: int) -> np.ndarray:
    """Prefer edge band + upper portrait area for portrait-style frames."""
    roi = np.zeros((h, w), dtype=bool)
    if scan_output.edge_mask is not None and scan_output.edge_mask.any():
        roi |= scan_output.edge_mask
    if scan_output.foreground_mask is not None and scan_output.foreground_mask.any():
        fg = scan_output.foreground_mask
        ys, _ = np.where(fg)
        if len(ys):
            y_cut = min(int(np.percentile(ys, 75)) + 1, h)
            roi[:y_cut, :] |= fg[:y_cut, :]
    if not roi.any():
        roi[:] = True
    return roi


def compute_blockiness_score(gray: np.ndarray) -> float:
    """DCT block boundary energy ratio on 8px grid."""
    h, w = gray.shape
    if h < 16 or w < 16:
        return 1.0
    gray_f = gray.astype(np.float32)
    boundary_h = np.abs(gray_f[:, 8::8] - gray_f[:, np.clip(np.arange(8, w, 8) - 1, 0, w - 1)])
    boundary_v = np.abs(gray_f[8::8, :] - gray_f[np.clip(np.arange(8, h, 8) - 1, 0, h - 1), :])
    ref_h = np.abs(gray_f[:, 4::8] - gray_f[:, np.clip(np.arange(4, w, 8) - 1, 0, w - 1)])
    ref_v = np.abs(gray_f[4::8, :] - gray_f[np.clip(np.arange(4, h, 8) - 1, 0, h - 1), :])
    e_h = float(boundary_h.mean()) if boundary_h.size else 0.0
    e_v = float(boundary_v.mean()) if boundary_v.size else 0.0
    r_h = float(ref_h.mean()) if ref_h.size else 1.0
    r_v = float(ref_v.mean()) if ref_v.size else 1.0
    e_ref = (r_h + r_v) / 2.0 + 1e-6
    return (e_h + e_v) / (2.0 * e_ref)


def compute_texture_loss_score(gray: np.ndarray, reference_var: float = 2400.0) -> float:
    """Rises when high-frequency energy drops (blur / compression_hf).

    注意：reference_var=2400 是以真实摄影/视频图像为基准校准的。
    AI 合成图（扩散模型等）天然具有低 Laplacian 方差的"柔和"风格，
    对此类图像请先用 is_ai_generated_style() 检查，再决定是否信任本分数。
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    if reference_var <= 0:
        return 0.0
    return max(0.0, (reference_var - var) / reference_var)


def is_ai_generated_style(gray: np.ndarray, lap_var_threshold: float = 80.0) -> bool:
    """启发式判断图像是否具有 AI 合成风格（低高频纹理 + 高平滑度）。

    AI 扩散模型生成的图像通常具有以下特征：
    1. 全图 Laplacian 方差极低（天然柔和，不是压缩失真导致）
    2. 像素值分布平滑，没有真实摄影中的高频噪声
    3. 灰度直方图呈平缓宽峰，方差较小

    这些特征会导致 compression_artifact / blur / mosaic 检测器产生大量误报，
    需要在检测前识别并做专项处理（提升阈值或标注为 generation_artifact）。

    Args:
        gray: 灰度图（uint8）
        lap_var_threshold: Laplacian 方差低于此值时认为具有 AI 合成风格。
            真实摄影图通常 > 200；AI 合成图通常 < 80。

    Returns:
        True 表示图像很可能是 AI 合成风格，应抑制基于纹理损失的误报。
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())
    if lap_var >= lap_var_threshold:
        return False

    # 进一步确认：检查梯度分布是否异常平滑
    # AI 合成图的梯度直方图峰值集中，真实图片梯度分布更分散
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_std = float(grad_mag.std())

    # 极低 Laplacian 方差 + 梯度标准差也很小 → 强 AI 合成信号
    # 若只有低 lap_var 但梯度标准差较大（如真实模糊图），不算 AI 风格
    return grad_std < 25.0


def foreground_mask_from_scan(scan_output: GlobalScanOutput, h: int, w: int) -> np.ndarray:
    if scan_output.foreground_mask is not None:
        return scan_output.foreground_mask
    return np.ones((h, w), dtype=bool)


def background_mask_from_scan(scan_output: GlobalScanOutput, h: int, w: int) -> np.ndarray:
    bg = ~foreground_mask_from_scan(scan_output, h, w)
    overlay = scan_output.overlay_mask
    if overlay is not None and overlay.any():
        bg &= ~overlay
    return bg


def face_region_mask(foreground: np.ndarray, h: int, w: int) -> np.ndarray:
    x, y, bw, bh = bbox_from_mask(foreground)
    if bw <= 0 or bh <= 0:
        return np.zeros((h, w), dtype=bool)
    fh = max(8, int(bh * 0.42))
    fw = max(8, int(bw * 0.55))
    cx = x + bw // 2
    cy = y + int(bh * 0.28)
    x0 = max(0, cx - fw // 2)
    y0 = max(0, cy - fh // 2)
    x1 = min(w, x0 + fw)
    y1 = min(h, y0 + fh)
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask & foreground


def hair_region_mask(foreground: np.ndarray, h: int, w: int) -> np.ndarray:
    x, y, bw, bh = bbox_from_mask(foreground)
    if bw <= 0 or bh <= 0:
        return np.zeros((h, w), dtype=bool)
    hh = max(8, int(bh * 0.35))
    mask = np.zeros((h, w), dtype=bool)
    mask[y : y + hh, x : x + bw] = True
    return mask & foreground


def hand_region_mask(foreground: np.ndarray, h: int, w: int) -> np.ndarray:
    x, y, bw, bh = bbox_from_mask(foreground)
    if bw <= 0 or bh <= 0:
        return np.zeros((h, w), dtype=bool)
    hw = max(8, int(bh * 0.45))
    y0 = y + int(bh * 0.45)
    y1 = min(h, y0 + hw)
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x : x + bw] = True
    return mask & foreground


def compute_mosaic_score(gray: np.ndarray) -> float:
    """Detect upscaled pixel blocks via down-up NEAREST consistency."""
    h, w = gray.shape
    if h < 32 or w < 32:
        return 0.0
    scale = 8
    small_h = max(4, h // scale)
    small_w = max(4, w // scale)
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    diff = float(np.abs(gray.astype(np.float32) - up.astype(np.float32)).mean())
    mosaic_likeness = float(np.clip(1.0 - diff / 18.0, 0, 1))

    bs = 8
    blocks_h = h // bs
    blocks_w = w // bs
    crop = gray[: blocks_h * bs, : blocks_w * bs].astype(np.float32)
    reshaped = crop.reshape(blocks_h, bs, blocks_w, bs).transpose(0, 2, 1, 3)
    flat_ratio = float((reshaped.var(axis=(2, 3)) < 12.0).mean())
    return float(np.clip(0.75 * mosaic_likeness + 0.25 * flat_ratio, 0, 1))


def localize_mosaic_mask(
    gray: np.ndarray,
    threshold: float,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    heat = _tile_heatmap(gray, compute_mosaic_score, tile_size=64, stride=32)
    return mask_from_heatmap(heat, threshold, roi_mask=roi_mask, min_area=200)


def compute_banding_score(gray: np.ndarray) -> float:
    """Horizontal color banding in smooth gradients."""
    profile = gray.mean(axis=1).astype(np.float32)
    if profile.size < 4 or float(profile.std()) < 3.0:
        return 0.0
    d1 = np.diff(profile)
    hist = np.histogram(d1, bins=32)[0]
    peak_ratio = float(hist.max() / max(d1.size, 1))
    step_ratio = float((np.abs(d1) > 2.5).mean())
    flat_ratio = float((np.abs(d1) < 0.4).mean())
    return float(np.clip(0.45 * peak_ratio + 0.35 * step_ratio + 0.2 * flat_ratio, 0, 1))


def localize_banding_mask(
    gray: np.ndarray,
    threshold: float,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    def _band_tile(patch: np.ndarray) -> float:
        return compute_banding_score(patch)

    heat = _tile_heatmap(gray, _band_tile, tile_size=96, stride=48)
    return mask_from_heatmap(heat, threshold, roi_mask=roi_mask, min_area=250)


def fft_highfreq_ratio(gray: np.ndarray) -> float:
    """Share of spectral energy outside low-frequency disk."""
    patch = gray.astype(np.float32)
    h, w = patch.shape
    if h < 16 or w < 16:
        return 1.0
    fshift = np.fft.fftshift(np.fft.fft2(patch))
    mag = np.abs(fshift)
    cy, cx = h // 2, w // 2
    radius = max(4, min(h, w) // 6)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    total = float(mag.sum()) + 1e-6
    outer = float(mag[dist > radius].sum())
    return outer / total


def blockiness_on_mask(gray: np.ndarray, mask: np.ndarray, pad: int = 8) -> float | None:
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y0, y1 = max(0, int(ys.min()) - pad), min(gray.shape[0], int(ys.max()) + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(gray.shape[1], int(xs.max()) + pad)
    crop = gray[y0:y1, x0:x1]
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    return compute_blockiness_score(crop)
