"""CLIP-IQA 无参考画质预测后端。

基于 `pyiqa <https://github.com/chaofengc/IQA-PyTorch>`_ 的 ``clipiqa`` metric
（CLIP-IQA, ICCV 2023）对整帧做无参考感知画质打分，再线性映射到 MOS [1, 5]。

依赖（可选，非核心流水线必需）::

    pip install "lqdd[clip_iqa]"
    # 即 pyiqa + torch + torchvision + setuptools

首次运行会自动下载 CLIP-IQA 权重（~260 MB，缓存到 ``~/.cache/torch/hub/pyiqa/``）。
"""

from __future__ import annotations

import numpy as np

# 模型懒加载单例：避免每帧都重新构建 / 重新下载权重。
_CLIP_IQA_METRIC = None
_CLIP_IQA_DEVICE = None


def _get_metric(device: str | None = None):
    """懒加载 pyiqa clipiqa metric。缺失依赖时抛 ImportError（由上层捕获降级）。"""
    global _CLIP_IQA_METRIC, _CLIP_IQA_DEVICE
    if _CLIP_IQA_METRIC is not None and (device is None or device == _CLIP_IQA_DEVICE):
        return _CLIP_IQA_METRIC

    import torch  # noqa: F401  — 提前给出清晰的 ImportError
    import pyiqa

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = pyiqa.create_metric("clipiqa", device=device)
    _CLIP_IQA_METRIC = metric
    _CLIP_IQA_DEVICE = device
    return metric


def _bgr_to_rgb_tensor(frame_bgr: np.ndarray, device: str = "cpu"):
    """BGR uint8 (H,W,3) → RGB float32 tensor (1,3,H,W) 0~1。"""
    import torch

    if frame_bgr is None:
        raise ValueError("frame_bgr is None")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"frame_bgr 必须是 (H,W,3) BGR uint8，实际 shape={frame_bgr.shape}")

    rgb = frame_bgr[:, :, ::-1].copy()  # BGR → RGB
    t = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def _score_to_mos(score: float, metric) -> float:
    """把 CLIP-IQA 原始分数线性映射到 MOS [1, 5]。

    CLIP-IQA 分数越高代表画质越好。score_range 通常为 [0, 1]；
    若 metric 暴露 ``score_range``，则先归一化到 [0, 1] 再映射到 [1, 5]。
    """
    lo, hi = 0.0, 1.0
    score_range = getattr(metric, "score_range", None)
    if score_range is not None:
        try:
            lo, hi = float(score_range[0]), float(score_range[1])
        except Exception:
            lo, hi = 0.0, 1.0
    span = hi - lo
    norm = (score - lo) / span if span > 0 else score
    norm = max(0.0, min(1.0, norm))
    mos = 1.0 + norm * 4.0
    return round(mos, 3)


def predict_mos_clip_iqa(frame_bgr: np.ndarray, device: str | None = None) -> float:
    """对单帧 BGR uint8 图像预测帧级 MOS（1~5）。

    Args:
        frame_bgr: ``(H, W, 3)`` BGR uint8 numpy 数组（OpenCV 原生格式）。
        device:    ``"cpu"`` / ``"cuda"``；None 时自动选择。

    Returns:
        MOS 分数（float，范围 [1, 5]，保留 3 位小数）。

    Raises:
        ImportError: pyiqa / torch 未安装（由 ``compute_mos`` 捕获后降级到 rule）。
        Exception:   推理失败（由 ``compute_mos`` 捕获后降级到 rule）。
    """
    import torch

    metric = _get_metric(device)
    dev = _CLIP_IQA_DEVICE or "cpu"
    img_t = _bgr_to_rgb_tensor(frame_bgr, device=dev)
    with torch.no_grad():
        score = metric(img_t)
    score_val = float(score.view(-1)[0].item())
    return _score_to_mos(score_val, metric)
