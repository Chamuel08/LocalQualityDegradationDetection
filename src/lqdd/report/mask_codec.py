from __future__ import annotations

import numpy as np


def encode_mask_rle(mask: np.ndarray) -> str:
    """Encode bool HxW mask as 'H,W:run1,run0,...' (row-major)."""
    flat = mask.astype(np.uint8).ravel()
    h, w = mask.shape
    if not flat.size:
        return f"{h},{w}:"
    runs: list[int] = []
    cur = int(flat[0])
    count = 1
    for v in flat[1:]:
        v = int(v)
        if v == cur:
            count += 1
        else:
            runs.append(count)
            cur = v
            count = 1
    runs.append(count)
    return f"{h},{w}:" + ",".join(str(x) for x in runs)


def decode_mask_rle(payload: str) -> np.ndarray:
    head, _, body = payload.partition(":")
    h_s, w_s = head.split(",")
    h, w = int(h_s), int(w_s)
    mask = np.zeros(h * w, dtype=bool)
    if not body:
        return mask.reshape(h, w)
    runs = [int(x) for x in body.split(",") if x]
    idx = 0
    val = False
    for run in runs:
        mask[idx : idx + run] = val
        idx += run
        val = not val
    return mask.reshape(h, w)
