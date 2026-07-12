"""Synthetic sample frames for demo and unit tests."""

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
    rng = np.random.default_rng(i + 100)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        base = 40 + int(8 * np.sin(y / 18.0 + i * 0.1))
        img[y, :] = (base, base + 35, base + 75)
    noise = rng.integers(-18, 19, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 4
    cv2.circle(img, (cx, cy), radius, (200, 180, 160), -1)
    face_noise = rng.integers(-22, 23, (h, w, 3), dtype=np.int16)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    face_mask = dist <= radius
    blended = img.astype(np.int16)
    blended[face_mask] = np.clip(blended[face_mask] + face_noise[face_mask], 0, 255)
    img = blended.astype(np.uint8)
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


def make_blur(i: int, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    img = make_normal(i, size)
    return cv2.GaussianBlur(img, (0, 0), sigmaX=5.0 + i * 0.2)


def make_mosaic(i: int, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    w, h = size
    tiny = cv2.resize(make_normal(i, (16, 12)), (w, h), interpolation=cv2.INTER_NEAREST)
    return tiny


def make_banding(i: int, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    w, h = size
    gradient = np.linspace(30, 220, h, dtype=np.uint8)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        val = int(gradient[y] // 12) * 12
        img[y, :] = (val, val // 2 + 20, 255 - val // 3)
    cv2.circle(img, (w // 2, h // 2 + 20), min(w, h) // 5, (200, 180, 160), -1)
    return img


def make_face_over(i: int, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    w, h = size
    img = make_normal(i, size)
    cx, cy = w // 2, h // 2 - 10
    cv2.circle(img, (cx, cy), min(w, h) // 6, (250, 250, 250), -1)
    return img


def make_hair_blur(i: int, size: tuple[int, int] = (320, 240)) -> np.ndarray:
    w, h = size
    img = make_normal(i, size)
    top = img[: h // 3, :].copy()
    img[: h // 3, :] = cv2.GaussianBlur(top, (0, 0), sigmaX=8.0)
    return img


def main() -> None:
    factories = [
        ("normal", make_normal),
        ("edge", make_green_edge),
        ("block", make_blocky),
        ("blur", make_blur),
        ("mosaic", make_mosaic),
        ("banding", make_banding),
        ("face_over", make_face_over),
        ("hair_blur", make_hair_blur),
    ]
    for category, factory in factories:
        for idx in range(1, 6):
            _save(OUT / category / f"{category}_{idx:02d}.png", factory(idx))
    _save(OUT / "edge" / "grey_edge_01.png", make_grey_edge(1))


if __name__ == "__main__":
    main()
