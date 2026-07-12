from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Callable, Literal

import cv2
import numpy as np

from lqdd.models.report import DegradationItem, QualityReport
from lqdd.report.mask_codec import decode_mask_rle

MaskVizStyle = Literal["contour_only", "contour_fill"]

DETECTOR_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "edge_bleed": (60, 76, 231),
    "compression_artifact": (0, 140, 255),
    "blur_artifact": (180, 120, 40),
    "mosaic_artifact": (200, 80, 180),
    "banding_artifact": (80, 180, 80),
    "background_artifact": (140, 140, 60),
    "hair_texture": (60, 180, 220),
    "face_artifact": (220, 100, 80),
    "hand_anomaly": (160, 80, 220),
}

DEGRADATION_LABELS: dict[str, str] = {
    "green_spill": "绿色溢色",
    "blockiness": "压缩伪影 / 纹理损失",
    "blur": "区域性模糊",
    "mosaic": "马赛克 / 像素块",
    "banding": "色带伪影",
    "background_artifact": "背景块效应",
    "hair_texture_loss": "发丝纹理损失",
    "face_artifact": "面部伪影",
    "overexposure": "面部过曝",
    "face_blur": "面部模糊",
    "hand_anomaly": "手部异常",
}

FILL_ALPHA = 0.28
CONTOUR_THICKNESS = 2


@dataclass(frozen=True)
class MaskVizVariant:
    key: MaskVizStyle
    title: str
    description: str


MASK_VIZ_VARIANTS: list[MaskVizVariant] = [
    MaskVizVariant(
        "contour_only",
        "轮廓线 only",
        "不规则 mask 彩色边缘，无填充、无图上文字；图例在图下方。",
    ),
    MaskVizVariant(
        "contour_fill",
        "轮廓 + 半透明填充",
        "mask 区域浅色填充 + 彩色轮廓；图例在图下方。",
    ),
]


def _color_for(deg: DegradationItem) -> tuple[int, int, int]:
    return DETECTOR_COLORS_BGR.get(deg.detector, (80, 80, 255))


def _legend_label(deg: DegradationItem) -> str:
    label = DEGRADATION_LABELS.get(deg.degradation_type, deg.degradation_type)
    return f"{label}（{deg.severity}）"


def _mask_for_degradation(deg: DegradationItem) -> np.ndarray | None:
    if not deg.region_mask_rle:
        return None
    mask = decode_mask_rle(deg.region_mask_rle)
    return mask if mask.any() else None


def _contours_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def render_mask_overlay(
    frame_bgr: np.ndarray,
    report: QualityReport,
    style: MaskVizStyle = "contour_fill",
) -> np.ndarray:
    if not report.degradations:
        return frame_bgr.copy()

    vis = frame_bgr.copy()
    for deg in report.degradations:
        mask = _mask_for_degradation(deg)
        if mask is None or not mask.any():
            continue
        color = _color_for(deg)
        if style == "contour_fill":
            colored = np.zeros_like(vis)
            colored[mask] = color
            vis = np.where(mask[:, :, None], cv2.addWeighted(colored, FILL_ALPHA, vis, 1 - FILL_ALPHA, 0), vis)
        for cnt in _contours_from_mask(mask):
            cv2.drawContours(vis, [cnt], -1, color, CONTOUR_THICKNESS, cv2.LINE_AA)
    return vis


def encode_image_b64(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_legend_html(degradations: list[DegradationItem]) -> str:
    if not degradations:
        return '<p class="legend-empty">未检出劣化，无需图例。</p>'
    seen: set[str] = set()
    chips: list[str] = []
    for deg in degradations:
        key = f"{deg.detector}:{deg.degradation_type}"
        if key in seen:
            continue
        seen.add(key)
        b, g, r = _color_for(deg)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        chips.append(
            f'<span class="legend-item">'
            f'<span class="swatch" style="background:{hex_color}"></span>'
            f"{_legend_label(deg)}"
            f"</span>"
        )
    return f'<div class="legend">{"".join(chips)}</div>'
