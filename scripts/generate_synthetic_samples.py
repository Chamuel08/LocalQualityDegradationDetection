"""Synthetic sample frames for v0.1 MVP tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample" / "frames"


def _save(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def make_normal(i: int, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (40, 80, 120)
    cv2.circle(img, (w // 2, h // 2), min(w, h) // 4, (200, 180, 160), -1)
    cv2.putText(img, f"normal_{i:02d}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def make_green_edge(i: int, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    w, h = size
    img = make_normal(i, size)
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 4
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    edge_band = (dist >= radius - 6) & (dist <= radius + 10)
    img[edge_band] = (20, 220, 20)
    return img


def make_grey_edge(i: int, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    w, h = size
    img = make_normal(i, size)
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 4
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    edge_band = (dist >= radius - 4) & (dist <= radius + 6)
    img[edge_band] = (40, 160, 40)
    return img


def make_blocky(i: int, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    w, h = size
    small = cv2.resize(make_normal(i, (40, 30)), (w, h), interpolation=cv2.INTER_NEAREST)
    noise = np.random.default_rng(i).integers(0, 8, (h, w, 3), dtype=np.uint8)
    return np.clip(small.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main() -> None:
    for category, factory in [
        ("normal", make_normal),
        ("edge", make_green_edge),
        ("block", make_blocky),
    ]:
        for idx in range(1, 6):
            _save(OUT / category / f"{category}_{idx:02d}.png", factory(idx))
    _save(OUT / "edge" / "grey_edge_01.png", make_grey_edge(1))


if __name__ == "__main__":
    main()
