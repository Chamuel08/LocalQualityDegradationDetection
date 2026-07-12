#!/usr/bin/env python3
"""
批量评测：合成 manifest GT vs lqdd / 基线。

用法（manifest 需自备，不随仓库分发）:
  .venv/bin/python benchmark/run_eval.py \\
    --manifest /path/to/manifest.json \\
    --output benchmark/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
for p in (str(REPO / "src"), str(BENCH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lqdd.config.loader import load_config
from lqdd.models.inputs import SingleFrameInput
from lqdd.pipeline.fast_pipeline import FastPipeline

from baselines import (
    baseline_global_blockiness,
    baseline_noop,
    baseline_oracle,
    baseline_random,
)
from metrics import best_iou, recall_at_iou, sample_detected
from type_mapping import GT_TYPE_TO_DETECTOR, is_supported_gt_type, pred_matches_gt_type


def load_manifest(path: Path) -> tuple[list[dict], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"] if isinstance(data, dict) else data
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    return samples, meta


def resolve_image(manifest_path: Path, sample: dict) -> Path:
    root = manifest_path.parent
    return (root / sample["image"]).resolve()


def gt_boxes(sample: dict) -> list[list[int]]:
    if sample.get("is_clean") or not sample.get("artifacts"):
        return []
    return [a["bbox"] for a in sample["artifacts"]]


def gt_types(sample: dict) -> list[str]:
    if sample.get("is_clean"):
        return []
    return [a["type"] for a in sample.get("artifacts", [])]


def run_lqdd(frame_bgr: np.ndarray, frame_id: str, pipeline: FastPipeline) -> list[dict]:
    inp = SingleFrameInput(frame=frame_bgr, frame_id=frame_id, mode="fast")
    report = pipeline.run(inp)
    out = []
    for d in report.degradations:
        out.append(
            {
                "detector": d.detector,
                "degradation_type": d.degradation_type,
                "bbox": list(d.bbox),
            }
        )
    return out


def filter_preds_for_gt(preds: list[dict], gt_type: str) -> list[list[int]]:
    if not gt_type:
        return [p["bbox"] for p in preds]
    return [
        p["bbox"]
        for p in preds
        if pred_matches_gt_type(p["detector"], p["degradation_type"], gt_type)
    ]


def eval_method_on_sample(
    method: str,
    sample: dict,
    frame_bgr: np.ndarray,
    pipeline: FastPipeline | None,
    rng: np.random.Generator,
) -> dict:
    gts = gt_boxes(sample)
    gt_type = gt_types(sample)[0] if gt_types(sample) else ""
    is_clean = sample.get("is_clean", False)

    if method == "noop":
        pred = baseline_noop(frame_bgr)
        pred_boxes = pred.boxes
    elif method == "global_blockiness":
        pred = baseline_global_blockiness(frame_bgr)
        pred_boxes = pred.boxes
    elif method == "random":
        pred = baseline_random(frame_bgr, rng, p_detect=0.5)
        pred_boxes = pred.boxes
    elif method == "oracle":
        pred = baseline_oracle(gts)
        pred_boxes = pred.boxes
    elif method == "lqdd":
        raw = run_lqdd(frame_bgr, sample["id"], pipeline)
        if is_clean:
            pred_boxes = [p["bbox"] for p in raw]
        else:
            pred_boxes = filter_preds_for_gt(raw, gt_type) if gt_type else [p["bbox"] for p in raw]
    else:
        raise ValueError(method)

    detected = sample_detected(pred_boxes)
    return {
        "method": method,
        "sample_id": sample["id"],
        "gt_type": gt_type,
        "is_clean": is_clean,
        "supported": is_supported_gt_type(gt_type) if gt_type else True,
        "detected": detected,
        "recall_iou_0.3": recall_at_iou(gts, pred_boxes, 0.3) if gts else None,
        "best_iou": best_iou(pred_boxes, gts[0]) if gts else None,
        "false_positive": is_clean and detected,
    }


def aggregate(rows: list[dict]) -> dict:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    summary = {}
    for method, items in by_method.items():
        pos = [x for x in items if not x["is_clean"]]
        neg = [x for x in items if x["is_clean"]]
        supported_pos = [x for x in pos if x["supported"]]
        block_pos = [x for x in pos if x.get("gt_type") == "block"]

        summary[method] = {
            "n_total": len(items),
            "n_positive": len(pos),
            "n_clean": len(neg),
            "detection_recall_supported": (
                sum(1 for x in supported_pos if x["detected"]) / len(supported_pos)
                if supported_pos
                else None
            ),
            "detection_recall_block": (
                sum(1 for x in block_pos if x["detected"]) / len(block_pos) if block_pos else None
            ),
            "recall_at_iou_0.3_supported": (
                sum(x["recall_iou_0.3"] for x in supported_pos if x["recall_iou_0.3"] is not None)
                / len(supported_pos)
                if supported_pos
                else None
            ),
            "fpr_on_clean": (sum(1 for x in neg if x["false_positive"]) / len(neg)) if neg else None,
            "unsupported_type_count": sum(1 for x in pos if not x["supported"]),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="合成劣化 manifest × lqdd 批量评测")
    parser.add_argument(
        "--manifest",
        required=True,
        help="评测 manifest.json 路径（需自备合成数据集，不随本仓库分发）",
    )
    parser.add_argument(
        "--output",
        default=str(REPO / "benchmark" / "results.json"),
        help="结果 JSON 输出路径（默认 benchmark/results.json，已 gitignore）",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"找不到 manifest: {manifest_path}")

    samples, meta = load_manifest(manifest_path)
    config = load_config(REPO / "config.yaml" if (REPO / "config.yaml").is_file() else None)
    pipeline = FastPipeline(config)
    rng = np.random.default_rng(args.seed)

    methods = ["noop", "global_blockiness", "random", "lqdd", "oracle"]
    rows: list[dict] = []

    for sample in samples:
        img_path = resolve_image(manifest_path, sample)
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"跳过无法读取: {img_path}", file=sys.stderr)
            continue
        for method in methods:
            rows.append(eval_method_on_sample(method, sample, frame, pipeline, rng))

    summary = aggregate(rows)
    out = {
        "manifest": str(manifest_path),
        "meta": meta,
        "methods": methods,
        "summary": summary,
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Evaluation Summary ===\n")
    print(f"{'方法':<20} {'block检出率':>10} {'支持类检出':>10} {'IoU@0.3':>10} {'干净误报':>10}")
    print("-" * 65)
    labels = {
        "noop": "B0 永远不检",
        "global_blockiness": "B1 整图block",
        "random": "B2 随机框",
        "lqdd": "B3 lqdd v0.1",
        "oracle": "Oracle 上界",
    }
    for m in methods:
        s = summary[m]
        print(
            f"{labels.get(m, m):<20} "
            f"{_pct(s['detection_recall_block']):>10} "
            f"{_pct(s['detection_recall_supported']):>10} "
            f"{_pct(s['recall_at_iou_0.3_supported']):>10} "
            f"{_pct(s['fpr_on_clean']):>10}"
        )
    print(f"\n详细结果: {out_path}")
    print("说明文档: benchmark/INTERVIEW_GUIDE.md")


def _pct(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.1f}%"


if __name__ == "__main__":
    main()
