"""Add matting (keying) artifacts at 3s & 7s + compression degradation to a
clean portrait video, preserving the original.

Matting artifacts injected around t=3s and t=7s (each a ~0.9s sin-window):
  - green spill: green tint blended into the person's edge/feather band
  - halo: a bright ring just outside the silhouette
  - jagged matte edge: block-quantized alpha -> sawtooth silhouette border
  - color fringing: R/B channels shifted outward a few px (chroma keying bleed)
  - semi-transparent ghost: a faint offset duplicate near the edge

Person mask comes from MediaPipe SelfieSegmentation (robust for portrait).

Compression: the final mp4 is encoded at a low bitrate so H.264 macroblocking
/ ringing appears across the whole clip (visible but not destroyed).

Usage:
    python scripts/degrade_jimeng_clip.py \
        --in  docs/demo/clip_jimeng_clean.mp4 \
        --out docs/demo/clip_jimeng_degraded.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

try:
    import mediapipe as mp  # noqa: F401  (optional, kept for future Tasks-API path)
    HAS_MP = True
except Exception:
    HAS_MP = False


def get_mask(segmenter, frame_bgr: np.ndarray) -> np.ndarray:
    """Soft person mask via grey-background keying.

    The clip is shot against a flat light-grey wall with a fixed camera, so the
    background is low-saturation + mid/high value. Everything else (skin, dark
    hair, red top, black pants) is more saturated or darker -> foreground.
    Returns a float32 mask in [0,1] (feathered).
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    # background = low saturation AND bright-ish (grey wall)
    bg = (s < 28) & (v > 90)
    fg = (~bg).astype(np.uint8) * 255
    # clean up: remove small holes / specks
    k = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    # keep the largest connected component (the person) to drop stray specks
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n_cc > 1:
        # skip background label 0; pick biggest remaining
        sizes = stats[1:, cv2.CC_STAT_AREA]
        biggest = 1 + int(np.argmax(sizes))
        fg = (labels == biggest).astype(np.uint8) * 255
    # feather the edge -> soft matte
    soft = cv2.GaussianBlur(fg.astype(np.float32), (11, 11), 0) / 255.0
    return soft


def _edge_bands(mask_soft: np.ndarray, mask_bin: np.ndarray):
    """Three thin rings around the silhouette (a few px wide), so artifacts
    NEVER cover the whole person -- only the matte edge."""
    k = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(mask_bin, k, iterations=1)
    dilated = cv2.dilate(mask_bin, k, iterations=2)
    inner = (mask_bin & (~eroded)).astype(np.float32)        # just inside border
    outer = (dilated & (~mask_bin)).astype(np.float32)       # just outside border
    feather = ((mask_soft > 0.15) & (mask_soft < 0.85)).astype(np.float32)  # soft transition
    return inner, outer, feather


def green_spill(frame: np.ndarray, mask_soft: np.ndarray, feather: np.ndarray,
                strength: float) -> np.ndarray:
    """Green tint ONLY in the soft matte transition band (peaks at mask=0.5)."""
    w = np.clip(1.0 - np.abs(mask_soft - 0.5) * 2.0, 0, 1)  # peaks at 0.5
    w = (w * feather * strength)
    out = frame.astype(np.float32)
    out[..., 1] += 60 * w        # green up
    out[..., 0] -= 12 * w        # B down
    out[..., 2] -= 12 * w        # R down
    return np.clip(out, 0, 255).astype(np.uint8)


def halo(frame: np.ndarray, outer: np.ndarray, strength: float) -> np.ndarray:
    """Bright ring ONLY just outside the silhouette."""
    ring = cv2.GaussianBlur(outer, (15, 15), 0) * strength
    out = frame.astype(np.float32) + 55 * ring[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def jagged_edge(frame: np.ndarray, mask_bin: np.ndarray, strength: float) -> np.ndarray:
    """Block-quantize the matte -> sawtooth border. Green is filled ONLY where the
    blocky matte differs from the smooth matte (the border notches), so the
    person interior is never greened -- only the jagged edge gets green notches."""
    h, w = mask_bin.shape
    block = 6
    small = cv2.resize(mask_bin.astype(np.float32), (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_NEAREST)
    blocky = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    notch = (np.abs(blocky - mask_bin.astype(np.float32)) > 0.5).astype(np.float32) * strength
    notch_blur = cv2.GaussianBlur(notch, (5, 5), 0)[..., None]
    green = np.array([0, 170, 40], np.float32)  # BGR green
    out = frame.astype(np.float32) * (1 - notch_blur * 0.85) + green * (notch_blur * 0.85)
    return np.clip(out, 0, 255).astype(np.uint8)


def color_fringing(frame: np.ndarray, mask_bin: np.ndarray, strength: float) -> np.ndarray:
    """Shift R and B channels outward along the edge -> chroma bleed (edge only)."""
    edge = cv2.Canny(mask_bin.astype(np.uint8) * 255, 50, 150)
    edge_dil = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
    band = (edge_dil > 0).astype(np.float32) * strength
    out = frame.astype(np.float32)
    shift = 4
    out[..., 2] = np.roll(out[..., 2], shift, axis=1)   # R right
    out[..., 0] = np.roll(out[..., 0], -shift, axis=1)  # B left
    res = frame.astype(np.float32) * (1 - band[..., None]) + out * band[..., None]
    return np.clip(res, 0, 255).astype(np.uint8)


def apply_matting(frame: np.ndarray, mask_soft: np.ndarray, strength: float) -> np.ndarray:
    """Apply all matting artifacts at the given strength in [0,1], edge-band only."""
    if strength <= 0:
        return frame
    mask_bin = (mask_soft > 0.5).astype(np.uint8)
    inner, outer, feather = _edge_bands(mask_soft, mask_bin)
    out = green_spill(frame, mask_soft, feather, strength)
    out = halo(out, outer, strength)
    out = jagged_edge(out, mask_bin, strength)
    out = color_fringing(out, mask_bin, strength)
    return out


def window_strength(t: float, centers: list[float], half_width: float) -> float:
    """Sin envelope: peaks at each center, 0 outside [center-half, center+half]."""
    s = 0.0
    for c in centers:
        d = abs(t - c)
        if d < half_width:
            s = max(s, np.cos(np.pi * d / (2 * half_width)) ** 2)
    return float(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="docs/demo/clip_jimeng_clean.mp4")
    ap.add_argument("--out", default="docs/demo/clip_jimeng_degraded.mp4")
    ap.add_argument("--matte-centers", default="3,7",
                    help="comma list of seconds where matting artifacts peak")
    ap.add_argument("--matte-half-width", type=float, default=0.45,
                    help="half-width of each matting window in seconds")
    ap.add_argument("--bitrate", default="900k",
                    help="final encode bitrate; lower = more compression artifacts")
    ap.add_argument("--max-side", type=int, default=720)
    args = ap.parse_args()

    centers = [float(x) for x in args.matte_centers.split(",")]

    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.inp}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"input: {W}x{H} @ {fps:.2f}fps, {n} frames")

    tmpdir = tempfile.mkdtemp(prefix="lqdd_deg_")

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        mask_soft = get_mask(None, frame)
        s = window_strength(t, centers, args.matte_half_width)
        out = apply_matting(frame, mask_soft, s)
        cv2.imwrite(os.path.join(tmpdir, f"f_{i:05d}.png"), out)
        if i % 30 == 0:
            print(f"  frame {i}/{n} t={t:.2f}s matte_strength={s:.2f}")
        i += 1
    cap.release()
    print(f"wrote {i} processed frames")

    out_abs = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    # combine processed frames + original audio, encode at low bitrate
    cmd = [
        "ffmpeg", "-y",
        "-framerate", f"{fps}",
        "-i", os.path.join(tmpdir, "f_%05d.png"),
        "-i", args.inp,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264",
        "-b:v", args.bitrate,
        "-maxrate", args.bitrate,
        "-bufsize", str(int(args.bitrate[:-1]) * 2) + "k",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        out_abs,
    ]
    print("ffmpeg:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2500:], file=sys.stderr)
        return r.returncode
    print(f"\nDONE -> {out_abs}  ({os.path.getsize(out_abs)/1024:.1f} KB)")
    print(f"original kept at: {os.path.abspath(args.inp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
