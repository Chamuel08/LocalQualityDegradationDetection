"""评测基线：每种基线回答一个对照问题。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lqdd.detectors.base import bbox_from_mask, clip_bbox, compute_blockiness_score


@dataclass
class PredResult:
    name: str
    boxes: list[list[int]]  # [x,y,w,h]
    detected: bool
    score: float | None = None
    detail: str = ""


def baseline_noop(_frame_bgr: np.ndarray) -> PredResult:
    """B0：永远不检出。下界——证明指标不是随便就能高的。"""
    return PredResult(name="noop", boxes=[], detected=False, detail="永远返回空")


def baseline_global_blockiness(frame_bgr: np.ndarray, threshold: float = 1.8) -> PredResult:
    """
    B1：全局 blockiness（简化版 Global-IQA 对照）。

    原理：和 lqdd compression_artifact 用同一套 DCT 8×8 网格边界能量，
    但故意做成「最笨」用法——分数高就认为整帧有问题，框=整图。
    用来证明：「只靠整图分数、不会局部定位」不够。
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    score = compute_blockiness_score(gray)
    if score < threshold:
        return PredResult(name="global_blockiness", boxes=[], detected=False, score=score)
    bbox = list(clip_bbox(bbox_from_mask(np.ones((h, w), dtype=bool)), w, h))
    return PredResult(
        name="global_blockiness",
        boxes=[bbox],
        detected=True,
        score=score,
        detail=f"整帧 blockiness={score:.2f} >= {threshold} → 框整图",
    )


def baseline_random(frame_bgr: np.ndarray, rng: np.random.Generator, p_detect: float = 0.5) -> PredResult:
    """B2：随机框。证明 localization 需要真方法，不是蒙也能高分。"""
    h, w = frame_bgr.shape[:2]
    if rng.random() > p_detect:
        return PredResult(name="random", boxes=[], detected=False)
    rw = int(rng.integers(max(32, w // 8), max(33, w // 2)))
    rh = int(rng.integers(max(32, h // 8), max(33, h // 2)))
    x = int(rng.integers(0, max(1, w - rw)))
    y = int(rng.integers(0, max(1, h - rh)))
    return PredResult(name="random", boxes=[[x, y, rw, rh]], detected=True, detail="随机框")


def baseline_oracle(gt_boxes: list[list[int]]) -> PredResult:
    """Oracle：直接读 GT 框。上界——指标和代码没写错时应该接近 100%。"""
    if not gt_boxes:
        return PredResult(name="oracle", boxes=[], detected=False, detail="GT 为空")
    return PredResult(
        name="oracle",
        boxes=[list(b) for b in gt_boxes],
        detected=True,
        detail="作弊：直接用 GT bbox",
    )
