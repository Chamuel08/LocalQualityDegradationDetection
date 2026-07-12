"""评测指标：IoU、Recall、FPR 等。"""

from __future__ import annotations


def iou_box(a: list[int], b: list[int]) -> float:
    """a, b: [x, y, w, h]"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def best_iou(pred_boxes: list[list[int]], gt_box: list[int]) -> float:
    if not pred_boxes:
        return 0.0
    return max(iou_box(p, gt_box) for p in pred_boxes)


def recall_at_iou(
    gt_boxes: list[list[int]],
    pred_boxes: list[list[int]],
    threshold: float = 0.3,
) -> float:
    if not gt_boxes:
        return 1.0
    hit = sum(1 for gt in gt_boxes if best_iou(pred_boxes, gt) >= threshold)
    return hit / len(gt_boxes)


def sample_detected(pred_boxes: list[list[int]]) -> bool:
    return len(pred_boxes) > 0
