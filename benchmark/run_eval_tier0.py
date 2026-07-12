#!/usr/bin/env python3
"""在 lqdd 自带 sample 上跑 Tier0 对齐评测（block / edge / normal）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
for p in (str(REPO / "src"), str(BENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lqdd.config.loader import load_config
from lqdd.models.inputs import SingleFrameInput
from lqdd.pipeline.fast_pipeline import FastPipeline


def expect_detector(category: str) -> str | None:
    if category == "block":
        return "compression_artifact"
    if category == "edge":
        return "edge_bleed"
    return None


def main() -> None:
    sample_root = REPO / "data" / "sample" / "frames"
    config = load_config(REPO / "config.yaml" if (REPO / "config.yaml").is_file() else None)
    pipe = FastPipeline(config)

    rows = []
    for category in ("block", "edge", "normal"):
        cat_dir = sample_root / category
        if not cat_dir.is_dir():
            continue
        exp = expect_detector(category)
        for img_path in sorted(cat_dir.glob("*.png")):
            frame = cv2.imread(str(img_path))
            report = pipe.run(SingleFrameInput(frame=frame, frame_id=img_path.stem))
            dets = {d.detector for d in report.degradations}
            hit = exp in dets if exp else len(dets) == 0
            rows.append(
                {
                    "path": str(img_path.relative_to(REPO)),
                    "category": category,
                    "expected": exp,
                    "detectors": sorted(dets),
                    "hit": hit,
                }
            )

    block_hits = [r for r in rows if r["category"] == "block" and r["hit"]]
    edge_hits = [r for r in rows if r["category"] == "edge" and r["hit"]]
    normal_ok = [r for r in rows if r["category"] == "normal" and r["hit"]]
    n_block = sum(1 for r in rows if r["category"] == "block")
    n_edge = sum(1 for r in rows if r["category"] == "edge")
    n_normal = sum(1 for r in rows if r["category"] == "normal")

    print("\n=== Tier0：lqdd 自带 sample（能力对齐集）===\n")
    for r in rows:
        mark = "OK" if r["hit"] else "MISS"
        print(f"  [{mark}] {r['path']} -> {r['detectors']}")

    print("\n汇总:")
    print(f"  block recall:  {len(block_hits)}/{n_block}")
    print(f"  edge recall:   {len(edge_hits)}/{n_edge}")
    print(f"  normal 正确:   {len(normal_ok)}/{n_normal} (无检出=正确)")

    out = BENCH / "results_tier0.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入: {out}")


if __name__ == "__main__":
    main()
