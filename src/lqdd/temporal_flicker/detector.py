from __future__ import annotations

"""TemporalFlicker — 时序闪烁检测器（V3 算法深化版）。

本检测器不进入 ALL_DETECTOR_NAMES 和 build_detector_registry，
因为它依赖多帧输入，无法在单帧 pipeline 中被调用。
正确的调用方式是通过 VideoClipRunner 在帧间聚合层使用。

算法（时序建模 + 局部画质劣化检测深化）：
1. 全局信号：相邻帧灰度均值差（亮度跳变）+ HSV-H 通道均值差（色彩漂移）
2. 运动补偿时序变化（C1）：相邻帧光流对齐后残差能量，区分真闪烁 vs 镜头运动
3. 时序 SSIM（C2）：相邻（运动补偿后）帧 SSIM，衡量时序一致性
4. 局部闪烁热力图（C3）：分块时序变化 → 热力图 + 带 bbox 的局部闪烁段
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FlickerSegment:
    """一段闪烁区间的描述。"""
    start_frame: int
    end_frame: int
    max_delta: float
    metric: str          # "luma_delta" | "hue_delta" | "motion_compensated_delta"
    severity: str        # "minor" | "moderate" | "critical"
    # C3 局部闪烁：可选 bbox（x, y, w, h），仅 localized_segments 填充
    bbox: list[int] | None = None


@dataclass
class TemporalFlickerResult:
    """对一组连续帧的时域闪烁检测结果。"""
    frame_count: int
    flicker_segments: list[FlickerSegment]
    mean_luma_delta: float
    max_luma_delta: float
    flicker_ratio: float          # 发生闪烁的帧间比例 [0, 1]
    is_flickering: bool
    method: str = "temporal_luma_hue_delta"
    # C1 运动补偿时序变化
    mean_motion_compensated_delta: float = 0.0
    max_motion_compensated_delta: float = 0.0
    # C2 时序 SSIM（相邻补偿帧 SSIM 均值，越高越一致）
    temporal_ssim: float = 1.0
    # C3 局部闪烁热力图（HxW float，每像素跨帧平均时序变化）
    flicker_heatmap: np.ndarray | None = None
    # C3 局部闪烁段（带 bbox）
    localized_segments: list[FlickerSegment] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 辅助：SSIM（uniform window 近似，Wang et al. 2004 的简化版）
# ---------------------------------------------------------------------------
_SSIM_C1 = 6.5025   # (0.01*255)^2
_SSIM_C2 = 58.5225   # (0.03*255)^2


def _ssim(x: np.ndarray, y: np.ndarray, win: int = 11) -> float:
    """两灰度图（uint8 0-255）的 SSIM 标量。"""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mu_x = cv2.blur(x, (win, win))
    mu_y = cv2.blur(y, (win, win))
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x_sq = cv2.blur(x * x, (win, win)) - mu_x_sq
    sigma_y_sq = cv2.blur(y * y, (win, win)) - mu_y_sq
    sigma_xy = cv2.blur(x * y, (win, win)) - mu_xy
    denom = (mu_x_sq + mu_y_sq + _SSIM_C1) * (sigma_x_sq + sigma_y_sq + _SSIM_C2)
    numer = (2 * mu_xy + _SSIM_C1) * (2 * sigma_xy + _SSIM_C2)
    ssim_map = numer / denom
    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# 辅助：光流对齐（把 next warp 回 prev 坐标）
# ---------------------------------------------------------------------------
def _warp_to_prev(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    """用 Farneback 光流把 next 对齐到 prev 坐标，返回 warped next。"""
    h, w = prev_gray.shape
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y + flow[:, :, 1]).astype(np.float32)
    warped = cv2.remap(
        next_gray, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def _severity_for(delta: float, threshold: float) -> str:
    if delta >= threshold * 2.5:
        return "critical"
    if delta >= threshold * 1.5:
        return "moderate"
    return "minor"


def detect_temporal_flicker(
    frames: list[np.ndarray],
    luma_delta_threshold: float = 8.0,
    hue_delta_threshold: float = 6.0,
    min_flicker_ratio: float = 0.15,
    motion_delta_threshold: float | None = None,
    enable_motion_compensation: bool = True,
    block_size: int = 16,
) -> TemporalFlickerResult:
    """对一组连续 BGR 帧检测时域闪烁（V3 算法深化版）。

    Args:
        frames: 连续帧列表（BGR uint8），至少 2 帧。
        luma_delta_threshold: 相邻帧灰度均值差阈值（0-255）。
        hue_delta_threshold: 相邻帧 H 通道均值差阈值（0-180）。
        min_flicker_ratio: 闪烁帧间比例超过此值才触发 is_flickering=True。
        motion_delta_threshold: 运动补偿残差能量阈值；None 时取 luma_delta_threshold。
        enable_motion_compensation: 是否启用 C1 光流运动补偿（关则只算 luma/hue）。
        block_size: C3 局部闪烁热力图的分块尺寸（px）；<=0 则不生成热力图。

    Returns:
        TemporalFlickerResult
    """
    n = len(frames)
    if n < 2:
        return TemporalFlickerResult(
            frame_count=n,
            flicker_segments=[],
            mean_luma_delta=0.0,
            max_luma_delta=0.0,
            flicker_ratio=0.0,
            is_flickering=False,
        )

    if motion_delta_threshold is None:
        motion_delta_threshold = luma_delta_threshold

    # 预计算每帧的灰度与 HSV-H
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    hues = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 0] for f in frames]

    luma_means = [float(g.mean()) for g in grays]
    hue_means = [float(h.mean()) for h in hues]

    luma_deltas = [abs(luma_means[i + 1] - luma_means[i]) for i in range(n - 1)]
    hue_deltas = [abs(hue_means[i + 1] - hue_means[i]) for i in range(n - 1)]

    flicker_segments: list[FlickerSegment] = []
    flicker_frame_count = 0

    # C1/C2 信号
    mc_deltas: list[float] = []
    ssim_vals: list[float] = []
    # C3 热力图：累加每对相邻帧的（运动补偿后）残差
    heatmap_accum: np.ndarray | None = None

    for i in range(n - 1):
        prev_gray = grays[i]
        next_gray = grays[i + 1]

        # C1 运动补偿残差 + C2 时序 SSIM
        if enable_motion_compensation:
            warped = _warp_to_prev(prev_gray, next_gray)
            residual = cv2.absdiff(prev_gray, warped)
            mc_delta = float(residual.mean())
            mc_deltas.append(mc_delta)
            ssim_vals.append(_ssim(prev_gray, warped))
            if block_size > 0:
                if heatmap_accum is None:
                    heatmap_accum = np.zeros_like(prev_gray, dtype=np.float64)
                heatmap_accum += residual.astype(np.float64)
        else:
            mc_deltas.append(0.0)
            ssim_vals.append(1.0)

        # 触发判定：luma / hue / motion_compensated 任一超阈值
        triggered = False
        metric = ""
        delta = 0.0
        threshold = luma_delta_threshold

        candidates = [
            ("luma_delta", luma_deltas[i], luma_delta_threshold),
            ("hue_delta", hue_deltas[i], hue_delta_threshold),
        ]
        if enable_motion_compensation:
            candidates.append(("motion_compensated_delta", mc_deltas[i], motion_delta_threshold))

        # 取相对阈值超得最多的那个 metric（同分时优先 luma_delta，更可解释）
        best_ratio = 0.0
        for m, d, thr in candidates:
            if d >= thr:
                ratio = d / thr if thr > 0 else 0.0
                if ratio > best_ratio:
                    best_ratio = ratio
                    triggered = True
                    metric = m
                    delta = d
                    threshold = thr

        if triggered:
            flicker_frame_count += 1
            flicker_segments.append(
                FlickerSegment(
                    start_frame=i,
                    end_frame=i + 1,
                    max_delta=round(delta, 3),
                    metric=metric,
                    severity=_severity_for(delta, threshold),
                )
            )

    mean_ld = float(np.mean(luma_deltas)) if luma_deltas else 0.0
    max_ld = float(max(luma_deltas)) if luma_deltas else 0.0
    flicker_ratio = flicker_frame_count / (n - 1) if n > 1 else 0.0

    mean_mc = float(np.mean(mc_deltas)) if mc_deltas else 0.0
    max_mc = float(max(mc_deltas)) if mc_deltas else 0.0
    mean_ssim = float(np.mean(ssim_vals)) if ssim_vals else 1.0

    # C3 局部闪烁热力图 + 局部段
    heatmap: np.ndarray | None = None
    localized_segments: list[FlickerSegment] = []
    if heatmap_accum is not None and block_size > 0:
        heatmap = (heatmap_accum / (n - 1)).astype(np.float32)
        localized_segments = _localized_segments(
            heatmap, block_size, motion_delta_threshold
        )

    return TemporalFlickerResult(
        frame_count=n,
        flicker_segments=flicker_segments,
        mean_luma_delta=round(mean_ld, 3),
        max_luma_delta=round(max_ld, 3),
        flicker_ratio=round(flicker_ratio, 3),
        is_flickering=(flicker_ratio >= min_flicker_ratio),
        method="temporal_luma_hue_motion_ssim",
        mean_motion_compensated_delta=round(mean_mc, 3),
        max_motion_compensated_delta=round(max_mc, 3),
        temporal_ssim=round(mean_ssim, 4),
        flicker_heatmap=heatmap,
        localized_segments=localized_segments,
    )


def _localized_segments(
    heatmap: np.ndarray,
    block_size: int,
    threshold: float,
) -> list[FlickerSegment]:
    """从热力图分块阈值化 + 连通域，产出带 bbox 的局部闪烁段。"""
    h, w = heatmap.shape
    ny, nx = max(1, h // block_size), max(1, w // block_size)
    block_avg = np.zeros((ny, nx), dtype=np.float32)
    for by in range(ny):
        for bx in range(nx):
            y0, y1 = by * block_size, min((by + 1) * block_size, h)
            x0, x1 = bx * block_size, min((bx + 1) * block_size, w)
            block_avg[by, bx] = float(heatmap[y0:y1, x0:x1].mean())

    block_mask = (block_avg >= threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(block_mask, connectivity=8)
    segments: list[FlickerSegment] = []
    for lbl in range(1, n_labels):
        bx, by, bw, bh, _area = stats[lbl]
        x0 = int(bx * block_size)
        y0 = int(by * block_size)
        x1 = int(min((bx + bw) * block_size, w))
        y1 = int(min((by + bh) * block_size, h))
        delta = float(block_avg[labels == lbl].max())
        segments.append(
            FlickerSegment(
                start_frame=-1,
                end_frame=-1,
                max_delta=round(delta, 3),
                metric="motion_compensated_delta",
                severity=_severity_for(delta, threshold),
                bbox=[x0, y0, x1 - x0, y1 - y0],
            )
        )
    return segments
