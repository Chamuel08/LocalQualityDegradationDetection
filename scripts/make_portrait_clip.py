"""Generate a portrait demo clip with REALISTIC compression artifacts + visible
motion + subtle flicker + occasional light face blur, for V2 video-clip demo.

Design choices (different from the previous over-aggressive version):
  - Motion is clearly visible: 20% slow zoom-in (1.0 -> 1.20) plus a horizontal
    sinusoidal pan, so the viewer obviously sees the camera moving. This gives
    optical-flow based temporal_flicker / temporal-SSIM detectors real motion
    to align against.
  - Compression artifacts are produced by ENCODING the clean frames with ffmpeg
    at a LOW bitrate (realistic streaming quality), NOT by Python block
    quantization. This yields natural H.264 macroblocking / ringing on edges
    and flat areas instead of an obviously synthetic blocky mess.
  - Temporal flicker: a tiny per-frame luma modulation (+/-3 levels) on flat
    areas, mimicking variable-bitrate streaming qp swings.
  - Face blur: light kernel (k=17), only on ~1/4 of frames, feathered, so it is
    a subtle "soft face" not a destroyed face.

Usage:
    python scripts/make_portrait_clip.py \
        --base docs/demo/synthetic_portrait.png \
        --out docs/demo/clip_portrait_degraded.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


def downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def zoom_pan(img: np.ndarray, t: float, zoom_end: float = 1.20) -> np.ndarray:
    """t in [0,1]. Visible zoom-in (1.0 -> zoom_end) + horizontal sinusoidal pan."""
    h, w = img.shape[:2]
    scale = 1.0 + (zoom_end - 1.0) * t
    new_w, new_h = int(w * scale), int(h * scale)
    big = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # horizontal pan oscillates: start centered, drift left then back
    pan_x = int((new_w - w) * (0.5 + 0.25 * np.sin(np.pi * t)))
    pan_y = int((new_h - h) * 0.5)
    pan_x = max(0, min(pan_x, new_w - w))
    pan_y = max(0, min(pan_y, new_h - h))
    return big[pan_y:pan_y + h, pan_x:pan_x + w].copy()


def detect_face_box(img: np.ndarray) -> tuple[int, int, int, int] | None:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(cascade_path):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return int(x), int(y), int(x + w), int(y + h)
    h, w = img.shape[:2]
    return (int(w * 0.32), int(h * 0.08), int(w * 0.68), int(h * 0.42))


def blur_region(img: np.ndarray, box: tuple[int, int, int, int], k: int = 17) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    pad = 10
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    mask[pad:y2 - y1 - pad, pad:x2 - x1 - pad] = 255
    mask = cv2.GaussianBlur(mask, (pad * 2 + 1, pad * 2 + 1), 0)
    mask_f = (mask.astype(np.float32) / 255.0)[..., None]
    out = img.copy()
    out[y1:y2, x1:x2] = (roi.astype(np.float32) * (1 - mask_f) + blurred.astype(np.float32) * mask_f).astype(np.uint8)
    return out


def add_subtle_flicker(img: np.ndarray, i: int, amount: float = 3.0) -> np.ndarray:
    """Tiny per-frame luma modulation on flat (low-variance) regions only."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # flatness mask: low local variance
    ksize = 16
    mean = cv2.blur(gray.astype(np.float32), (ksize, ksize))
    sq = cv2.blur((gray.astype(np.float32)) ** 2, (ksize, ksize))
    var = sq - mean ** 2
    flat_mask = (var < 25).astype(np.float32)  # roughly flat areas
    delta = amount * np.sin(2 * np.pi * i / 6.0)  # period ~6 frames
    out = img.astype(np.float32)
    out[..., 0] += delta * flat_mask
    out[..., 1] += delta * flat_mask
    out[..., 2] += delta * flat_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="docs/demo/synthetic_portrait.png")
    ap.add_argument("--out", default="docs/demo/clip_portrait_degraded.mp4")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--max-side", type=int, default=720)
    ap.add_argument("--zoom-end", type=float, default=1.20)
    ap.add_argument("--bitrate", type=str, default="700k",
                    help="ffmpeg target bitrate; lower = more compression artifacts")
    ap.add_argument("--face-blur-every", type=int, default=5,
                    help="blur face every N-th frame (0 disables)")
    ap.add_argument("--face-blur-offset", type=int, default=2,
                    help="skip face blur until this frame (keep first frames clean)")
    ap.add_argument("--face-blur-k", type=int, default=13,
                    help="face blur kernel size (smaller = subtler)")
    args = ap.parse_args()

    base = cv2.imread(args.base)
    if base is None:
        print(f"ERROR: cannot read {args.base}", file=sys.stderr)
        return 1
    base = downscale(base, args.max_side)
    H, W = base.shape[:2]
    print(f"base frame: {W}x{H}, {args.frames} frames @ {args.fps} fps, "
          f"zoom 1.0->{args.zoom_end}, bitrate {args.bitrate}")

    face_box = detect_face_box(base)
    print(f"face box: {face_box}")

    tmpdir = tempfile.mkdtemp(prefix="lqdd_clip_")
    for i in range(args.frames):
        t = i / max(1, args.frames - 1)
        frame = zoom_pan(base, t, zoom_end=args.zoom_end)

        # subtle temporal flicker on flat areas (very small delta)
        frame = add_subtle_flicker(frame, i, amount=3.0)

        # occasional light face blur (skip first frames so the clip opens clean)
        do_blur = (face_box is not None and args.face_blur_every
                   and i >= args.face_blur_offset
                   and (i - args.face_blur_offset) % args.face_blur_every == 0)
        if do_blur:
            fb = detect_face_box(frame) or face_box
            frame = blur_region(frame, fb, k=args.face_blur_k)

        cv2.imwrite(os.path.join(tmpdir, f"f_{i:04d}.png"), frame)
        if i % 10 == 0:
            print(f"  frame {i}/{args.frames} face_blur={'yes' if do_blur else 'no'}")

    # encode clean-ish frames at LOW bitrate -> realistic H.264 compression artifacts
    out_abs = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", os.path.join(tmpdir, "f_%04d.png"),
        "-c:v", "libx264",
        "-b:v", args.bitrate,
        "-maxrate", args.bitrate,
        "-bufsize", str(int(args.bitrate[:-1]) * 2) + "k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_abs,
    ]
    print("ffmpeg:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
        return r.returncode
    print(f"\nDONE -> {out_abs}  ({os.path.getsize(out_abs)/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
