#!/usr/bin/env python3
"""Smoke eval on GT benchmark manifest (replaces Tier0 data/sample/frames)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
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

from type_mapping import GT_TYPE_TO_DETECTOR, pred_matches_gt_type


def default_manifest() -> Path:
    env = os.environ.get("LQDD_BENCHMARK_MANIFEST")
    if env:
        return Path(env)
    data = Path(os.environ.get("LQDD_DATA_DIR", Path.home() / "data"))
    return data / "synthetic_benchmark" / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke eval on GT benchmark manifest")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=BENCH / "results_smoke.json")
    args = parser.parse_args()

    manifest_path = (args.manifest or default_manifest()).resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = data.get("samples", data)
    root = manifest_path.parent

    config = load_config(REPO / "config.yaml" if (REPO / "config.yaml").is_file() else None)
    pipe = FastPipeline(config)

    rows = []
    for sample in samples:
        img_path = (root / sample["image"]).resolve()
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        report = pipe.run(SingleFrameInput(frame=frame, frame_id=sample.get("id", img_path.stem)))
        preds = [{"detector": d.detector, "degradation_type": d.degradation_type} for d in report.degradations]

        if sample.get("is_clean"):
            hit = len(preds) == 0
            gt_type = "clean"
        else:
            gt_type = sample.get("primary_type") or (sample.get("artifacts") or [{}])[0].get("type", "")
            expected = GT_TYPE_TO_DETECTOR.get(gt_type)
            hit = expected is not None and any(
                pred_matches_gt_type(p["detector"], p["degradation_type"], gt_type) for p in preds
            )

        rows.append(
            {
                "id": sample.get("id"),
                "path": str(img_path),
                "gt_type": gt_type,
                "expected_detector": GT_TYPE_TO_DETECTOR.get(gt_type),
                "detectors": [p["detector"] for p in preds],
                "hit": hit,
                "is_clean": sample.get("is_clean", False),
            }
        )

    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r["gt_type"]].append(r)

    print(f"\n=== Smoke eval: {manifest_path.name} ===\n")
    for gt_type, group in sorted(by_type.items()):
        hits = sum(1 for r in group if r["hit"])
        print(f"  {gt_type:<18} {hits}/{len(group)}")

    clean = [r for r in rows if r["is_clean"]]
    if clean:
        ok = sum(1 for r in clean if r["hit"])
        print(f"\n  clean (zero det): {ok}/{len(clean)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
