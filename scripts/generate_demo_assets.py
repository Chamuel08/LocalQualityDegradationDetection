#!/usr/bin/env python3
"""Generate demo assets under examples/ from GT benchmark manifest."""

from __future__ import annotations

import json
import os
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
DEFAULT_DATA = Path(os.environ.get("LQDD_DATA_DIR", Path.home() / "data"))
DEFAULT_MANIFEST = DEFAULT_DATA / "synthetic_benchmark" / "manifest.json"

TYPE_TITLES: dict[str, str] = {
    "edge_compression": "边缘压缩伪影（compression_artifact）",
    "block": "压缩块效应（compression_artifact）",
    "blur": "区域性模糊（blur_artifact）",
    "mosaic": "马赛克（mosaic_artifact）",
    "banding": "色带（banding_artifact）",
    "overexposure": "面部过曝（face_artifact）",
    "green_spill": "绿色溢色（edge_bleed）",
    "hair_texture": "发丝糊化（hair_texture）",
    "clean": "干净源图（负样本）",
}


def _benchmark_manifest() -> Path:
    env = os.environ.get("LQDD_BENCHMARK_MANIFEST")
    if env:
        return Path(env)
    if DEFAULT_MANIFEST.is_file():
        return DEFAULT_MANIFEST
    fallback = DEFAULT_DATA / "synthetic_benchmark" / "manifest.json"
    return fallback


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


def _summary_panel(report, width: int, meta: dict) -> np.ndarray:
    mos_val = report.overall_mos
    mos_str = f"{mos_val:.2f}" if mos_val is not None else "unavailable"
    lines = [
        f"overall_mos: {mos_str}",
        f"degradations: {len(report.degradations)}",
        f"gt_type: {meta.get('primary_type', '')}",
    ]
    if meta.get("source"):
        lines.append(f"source: {meta['source']}")
    for idx, deg in enumerate(report.degradations[:3], start=1):
        lines.append(f"{idx}. {deg.degradation_type} | {deg.severity} | conf={deg.confidence:.2f}")
        for wrapped in textwrap.wrap(deg.evidence.detail, width=68):
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


def _compose_side_by_side(input_bgr: np.ndarray, overlay_bgr: np.ndarray, report, meta: dict) -> np.ndarray:
    tile_w = 420
    left = _label_image(_resize_width(input_bgr, tile_w), "GT benchmark input")
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
    panel = _summary_panel(report, row.shape[1], meta)
    return cv2.vconcat([row, np.full((8, row.shape[1], 3), 255, dtype=np.uint8), panel])


def _pick_demo_cases(manifest_path: Path) -> list[dict]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    samples = data.get("samples", data)
    by_type: dict[str, dict] = {}
    for sample in samples:
        ptype = sample.get("primary_type")
        if not ptype:
            arts = sample.get("artifacts") or []
            ptype = arts[0]["type"] if arts else ("clean" if sample.get("is_clean") else "unknown")
        if ptype not in by_type:
            by_type[ptype] = sample
    order = [
        "green_spill",
        "edge_compression",
        "block",
        "blur",
        "mosaic",
        "banding",
        "overexposure",
        "hair_texture",
        "clean",
    ]
    picked = []
    for key in order:
        if key in by_type:
            picked.append(by_type[key])
    return picked, root


def main() -> None:
    manifest_path = _benchmark_manifest()
    if not manifest_path.is_file():
        raise SystemExit(
            f"Benchmark manifest not found: {manifest_path}\n"
            "Run: python scripts/generate_benchmark_dataset.py"
        )

    EXAMPLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    VIZ_STYLES.mkdir(parents=True, exist_ok=True)

    config = load_config(REPO / "config.example.yaml")
    config.report.system_version = "0.1.0"
    pipeline = FastPipeline(config)

    cases, root = _pick_demo_cases(manifest_path)
    manifest_out: list[dict] = []

    for sample in cases:
        ptype = sample.get("primary_type") or "unknown"
        key = ptype.replace("/", "_")
        title = TYPE_TITLES.get(ptype, ptype)
        image_path = (root / sample["image"]).resolve()
        frame = _load_image(image_path)
        frame_id = sample.get("id", image_path.stem)
        report = pipeline.run(SingleFrameInput(frame=frame, frame_id=frame_id, mode="fast"))
        overlay = render_mask_overlay(frame, report, style="contour_fill")

        html_path = REPORTS / f"{key}.html"
        render_html_report(report, html_path, frame_bgr=frame, viz_style="contour_fill")
        cv2.imwrite(str(REPORTS / f"{key}_overlay.png"), overlay)

        meta = {
            "primary_type": ptype,
            "source": sample.get("source"),
            "benchmark_id": sample.get("id"),
        }
        manifest_out.append(
            {
                "key": key,
                "title": title,
                "image": str(image_path),
                "html": str(html_path.relative_to(REPO)),
                "mos": round(report.overall_mos, 2) if report.overall_mos is not None else None,
                "degradations": len(report.degradations),
                "detectors": [d.detector for d in report.degradations],
                "gt_artifacts": sample.get("artifacts", []),
                **meta,
            }
        )

        if ptype in ("green_spill", "edge_compression") and not (EXAMPLES / "demo_report.html").exists():
            demo_input = EXAMPLES / "demo_input.png"
            cv2.imwrite(str(demo_input), frame)
            cv2.imwrite(str(EXAMPLES / "demo_report.png"), _compose_side_by_side(frame, overlay, report, meta))
            render_html_report(report, EXAMPLES / "demo_report.html", frame_bgr=frame)
            (EXAMPLES / "demo_report.json").write_text(
                json.dumps(
                    {
                        "input_image": str(image_path),
                        "benchmark_manifest": str(manifest_path),
                        "overall_mos": report.overall_mos,
                        "gt_primary_type": ptype,
                        "degradations": [
                            {
                                "type": d.degradation_type,
                                "detector": d.detector,
                                "severity": d.severity,
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

    block_case = next((c for c in cases if (c.get("primary_type") == "block")), None)
    if block_case:
        block_path = (root / block_case["image"]).resolve()
        block_frame = _load_image(block_path)
        block_report = pipeline.run(
            SingleFrameInput(frame=block_frame, frame_id=block_case.get("id", "block"), mode="fast")
        )
        cv2.imwrite(str(VIZ_STYLES / "contour_only.png"), render_mask_overlay(block_frame, block_report, style="contour_only"))
        cv2.imwrite(str(VIZ_STYLES / "contour_fill.png"), render_mask_overlay(block_frame, block_report, style="contour_fill"))

    (EXAMPLES / "manifest.json").write_text(
        json.dumps(
            {"cases": manifest_out, "benchmark_manifest": str(manifest_path)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {EXAMPLES / 'demo_report.html'}")
    print(f"Wrote {REPORTS}/")


if __name__ == "__main__":
    main()
