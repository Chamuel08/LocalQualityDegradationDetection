from __future__ import annotations

from typing import Protocol

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
