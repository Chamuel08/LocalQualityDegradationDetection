#!/usr/bin/env python3
"""Generate public demo assets under examples/ (repo-bundled sample frames only)."""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import cv2
import numpy as np

from lqdd.config.loader import load_config
from lqdd.models.inputs import SingleFrameInput
from lqdd.pipeline.fast_pipeline import FastPipeline
from lqdd.report.html_renderer import render_html_report
from lqdd.report.viz_variants import render_mask_overlay

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
REPORTS = EXAMPLES / "reports"
VIZ_STYLES = EXAMPLES / "viz_styles"
SAMPLES = REPO / "data" / "sample" / "frames"

DEMO_CASES: list[tuple[str, Path, str]] = [
    ("edge", SAMPLES / "edge" / "edge_01.png", "绿色溢色（edge_bleed）"),
    ("block", SAMPLES / "block" / "block_01.png", "压缩块效应（compression）"),
    ("blur", SAMPLES / "blur" / "blur_01.png", "区域性模糊（blur_artifact）"),
    ("mosaic", SAMPLES / "mosaic" / "mosaic_01.png", "马赛克（mosaic_artifact）"),
    ("banding", SAMPLES / "banding" / "banding_01.png", "色带（banding_artifact）"),
    ("face_over", SAMPLES / "face_over" / "face_over_01.png", "面部过曝（face_artifact）"),
    ("normal", SAMPLES / "normal" / "normal_01.png", "干净帧（无检出）"),
]


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise OSError(f"cannot read image: {path}")
    return img


def _resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    if frame.shape[1] == width:
        return frame
    ratio = width / frame.shape[1]
    height = max(1, int(frame.shape[0] * ratio))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _label_image(frame: np.ndarray, title: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (255, 255, 255), thickness=-1)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    return out


def _summary_panel(report, width: int) -> np.ndarray:
    lines = [
        f"overall_mos: {report.overall_mos:.2f}",
        f"degradations: {len(report.degradations)}",
    ]
    for idx, deg in enumerate(report.degradations[:2], start=1):
        ev = deg.evidence
        lines.append(f"{idx}. {deg.degradation_type} | {deg.severity} | conf={deg.confidence:.2f}")
        lines.append(f"   mask: {'yes' if deg.region_mask_rle else 'no'}")
        for wrapped in textwrap.wrap(ev.detail, width=68):
            lines.append(f"   {wrapped}")
    line_h = 22
    pad = 16
    height = pad * 2 + line_h * len(lines)
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    y = pad + 16
    for line in lines:
        cv2.putText(panel, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 44, 52), 1, cv2.LINE_AA)
        y += line_h
    return panel


def _compose_side_by_side(input_bgr: np.ndarray, overlay_bgr: np.ndarray, report) -> np.ndarray:
    tile_w = 420
    left = _label_image(_resize_width(input_bgr, tile_w), "Input")
    right = _label_image(_resize_width(overlay_bgr, tile_w), "Mask overlay")
    row_h = max(left.shape[0], right.shape[0])
    gap = 12

    def pad_h(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == row_h:
            return img
        canvas = np.full((row_h, img.shape[1], 3), 255, dtype=np.uint8)
        y0 = (row_h - img.shape[0]) // 2
        canvas[y0 : y0 + img.shape[0], : img.shape[1]] = img
        return canvas

    row = cv2.hconcat([pad_h(left), np.full((row_h, gap, 3), 255, dtype=np.uint8), pad_h(right)])
    panel = _summary_panel(report, row.shape[1])
    return cv2.vconcat([row, np.full((8, row.shape[1], 3), 255, dtype=np.uint8), panel])


def main() -> None:
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    VIZ_STYLES.mkdir(parents=True, exist_ok=True)

    config = load_config(REPO / "config.example.yaml")
    config.report.system_version = "0.1.0"
    pipeline = FastPipeline(config)

    manifest: list[dict] = []
    primary_overlay = None
    primary_input = None
    primary_report = None

    for key, image_path, title in DEMO_CASES:
        frame = _load_image(image_path)
        report = pipeline.run(SingleFrameInput(frame=frame, frame_id=image_path.stem, mode="fast"))
        overlay = render_mask_overlay(frame, report, style="contour_fill")

        html_path = REPORTS / f"{key}_01.html"
        render_html_report(report, html_path, frame_bgr=frame, viz_style="contour_fill")
        cv2.imwrite(str(REPORTS / f"{key}_01_overlay.png"), overlay)

        manifest.append(
            {
                "key": key,
                "title": title,
                "image": str(image_path.relative_to(REPO)),
                "html": str(html_path.relative_to(REPO)),
                "mos": round(report.overall_mos, 2),
                "degradations": len(report.degradations),
            }
        )

        if key == "edge":
            primary_input = frame
            primary_overlay = overlay
            primary_report = report
            shutil.copy2(image_path, EXAMPLES / "demo_input.png")
            cv2.imwrite(str(EXAMPLES / "demo_report.png"), _compose_side_by_side(frame, overlay, report))
            render_html_report(report, EXAMPLES / "demo_report.html", frame_bgr=frame)
            (EXAMPLES / "demo_report.json").write_text(
                json.dumps(
                    {
                        "overall_mos": report.overall_mos,
                        "degradation_count": len(report.degradations),
                        "degradations": [
                            {
                                "type": d.degradation_type,
                                "severity": d.severity,
                                "has_mask": bool(d.region_mask_rle),
                                "detail": d.evidence.detail,
                            }
                            for d in report.degradations
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    block_frame = _load_image(SAMPLES / "block" / "block_01.png")
    block_report = pipeline.run(SingleFrameInput(frame=block_frame, frame_id="block_viz", mode="fast"))
    cv2.imwrite(
        str(VIZ_STYLES / "contour_only.png"),
        render_mask_overlay(block_frame, block_report, style="contour_only"),
    )
    cv2.imwrite(
        str(VIZ_STYLES / "contour_fill.png"),
        render_mask_overlay(block_frame, block_report, style="contour_fill"),
    )

    (EXAMPLES / "manifest.json").write_text(
        json.dumps({"cases": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {EXAMPLES / 'demo_report.html'}")
    print(f"Wrote {REPORTS}/ (edge, block, normal)")
    print(f"Wrote {VIZ_STYLES}/ (contour_only, contour_fill)")


if __name__ == "__main__":
    main()
