#!/usr/bin/env python3
"""D2 VLM prompt 消融实验：评估不同 vlm_discover prompt 对检测率 / FPR 的影响。

在带 GT 的基准集上跑三种 prompt（baseline / strict / loose），统计：
  - TPR（检测率）= TP / (TP + FN)
  - FPR（误检率）= FP / (FP + TN)
  - 平均每帧 findings 数
  - 平均 VLM 延迟

用法：
  # 真实 VLM（需 ollama + qwen2.5-vl:7b + 基准集）
  python scripts/vlm_prompt_ablation.py --manifest ~/data/synthetic_benchmark/manifest.json

  # mock 模式（无需 VLM，用 MockVLMClient 演示流程，输出示例表）
  python scripts/vlm_prompt_ablation.py --mock --samples 20

输出：stdout 一张 markdown 表 + 可选 --out 写入 .md 文件。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lqdd.agent.prompts import VLM_DISCOVER_PROMPT_VARIANTS
from lqdd.config.loader import VLMConfig, load_config
from lqdd.vlm.client import OllamaVLMClient, VLMClient
from lqdd.vlm.confirm import build_vlm_client

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(os.environ.get("LQDD_DATA_DIR", Path.home() / "data"))
DEFAULT_MANIFEST = DEFAULT_DATA / "synthetic_benchmark" / "manifest.json"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """基准集单条样本。"""
    image_path: Path
    is_degraded: bool            # GT：是否存在劣化
    degradation_types: list[str]  # GT：劣化类型列表（clean 为 []）


@dataclass
class VariantResult:
    variant: str
    tpr: float
    fpr: float
    avg_findings: float
    avg_latency_ms: float
    n_samples: int
    n_pos: int
    n_neg: int


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_samples(manifest_path: Path, samples: int | None = None) -> list[Sample]:
    """从基准集 manifest 加载样本。

    manifest 格式（generate_benchmark_dataset.py 产出）：
      [
        {"path": "...", "label": "clean" | "<degradation_type>", "is_degraded": bool, ...},
        ...
      ]
    兼容字段名：image_path / path / file；is_degraded / degraded；degradation_types / types。
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest 不存在：{manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: list[Sample] = []
    base = manifest_path.parent
    for item in raw:
        p = item.get("image_path") or item.get("path") or item.get("file")
        if p is None:
            continue
        path = Path(p)
        if not path.is_absolute():
            path = base / path
        is_deg = bool(item.get("is_degraded", item.get("degraded", item.get("label") != "clean")))
        types = item.get("degradation_types") or item.get("types") or (
            [] if not is_deg else [item.get("label", "unknown")]
        )
        out.append(Sample(image_path=path, is_degraded=is_deg, degradation_types=list(types)))
    if samples is not None:
        out = out[:samples]
    return out


def generate_mock_samples(n: int) -> list[Sample]:
    """生成 mock 样本：一半劣化一半干净，图像为合成噪声。"""
    rng = np.random.default_rng(42)
    out: list[Sample] = []
    for i in range(n):
        is_deg = i < n // 2
        img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        if is_deg:
            # 注入一个明显的块状劣化区域
            img[40:80, 40:80] = rng.integers(200, 256, size=(40, 40, 3), dtype=np.uint8)
        path = Path(f"/tmp/lqdd_ablation_mock_{i}.png")
        cv2.imwrite(str(path), img)
        out.append(Sample(
            image_path=path,
            is_degraded=is_deg,
            degradation_types=["compression_artifact"] if is_deg else [],
        ))
    return out


# ---------------------------------------------------------------------------
# VLM 调用
# ---------------------------------------------------------------------------


def run_vlm_discover(
    vlm: VLMClient,
    prompt: str,
    image_path: Path,
) -> tuple[list[dict[str, Any]], float]:
    """对单张图执行 vlm_discover，返回 (findings, latency_ms)。"""
    img = cv2.imread(str(image_path))
    if img is None:
        return [], 0.0
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return [], 0.0
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    t0 = time.perf_counter()
    raw = vlm.confirm(prompt, b64)
    lat = (time.perf_counter() - t0) * 1000.0
    if raw is None:
        return [], lat
    try:
        data = raw if isinstance(raw, dict) else json.loads(str(raw))
        findings = data.get("findings", []) if isinstance(data, dict) else []
        return findings if isinstance(findings, list) else [], lat
    except Exception:
        return [], lat


# ---------------------------------------------------------------------------
# 评测指标
# ---------------------------------------------------------------------------


def evaluate_variant(
    variant: str,
    vlm: VLMClient,
    samples: list[Sample],
) -> VariantResult:
    """跑一种 prompt 在所有样本上，计算 TPR / FPR / 平均 findings / 平均延迟。

    判定规则（与 GT 对齐）：
      - VLM 输出 findings 非空 -> 预测为劣化
      - GT is_degraded=True 且预测劣化 -> TP
      - GT is_degraded=False 且预测劣化 -> FP
      - GT is_degraded=True 且预测干净 -> FN
      - GT is_degraded=False 且预测干净 -> TN
    """
    prompt = VLM_DISCOVER_PROMPT_VARIANTS[variant]
    tp = fp = fn = tn = 0
    total_findings = 0
    total_lat = 0.0
    for s in samples:
        findings, lat = run_vlm_discover(vlm, prompt, s.image_path)
        predicted_degraded = len(findings) > 0
        total_findings += len(findings)
        total_lat += lat
        if s.is_degraded and predicted_degraded:
            tp += 1
        elif s.is_degraded and not predicted_degraded:
            fn += 1
        elif (not s.is_degraded) and predicted_degraded:
            fp += 1
        else:
            tn += 1
    n_pos = tp + fn
    n_neg = fp + tn
    tpr = tp / n_pos if n_pos > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    n = len(samples)
    return VariantResult(
        variant=variant,
        tpr=round(tpr, 3),
        fpr=round(fpr, 3),
        avg_findings=round(total_findings / n, 3) if n > 0 else 0.0,
        avg_latency_ms=round(total_lat / n, 2) if n > 0 else 0.0,
        n_samples=n,
        n_pos=n_pos,
        n_neg=n_neg,
    )


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def render_markdown_table(results: list[VariantResult]) -> str:
    lines = [
        "# D2 VLM prompt 消融实验结果",
        "",
        "评估不同 `vlm_discover` prompt 变体对检测率（TPR）/ 误检率（FPR）的影响。",
        "",
        "| Variant | TPR (检测率) | FPR (误检率) | Avg Findings | Avg Latency (ms) | N (pos/neg) |",
        "|---------|--------------|--------------|-------------|------------------|-------------|",
    ]
    for r in results:
        lines.append(
            f"| {r.variant} | {r.tpr:.3f} | {r.fpr:.3f} | {r.avg_findings:.2f} | "
            f"{r.avg_latency_ms:.1f} | {r.n_samples} ({r.n_pos}/{r.n_neg}) |"
        )
    lines.append("")
    lines.append("**结论建议**：")
    lines.append("- `baseline`：生产默认，平衡召回与精度")
    lines.append("- `strict`：高置信度阈值，降低 FPR，适合误报敏感场景")
    lines.append("- `loose`：低阈值高召回，适合漏检敏感场景（如 AIGC 审核）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mock VLM（用于无 Ollama 环境下的流程演示）
# ---------------------------------------------------------------------------


class _MockAblationVLM(VLMClient):
    """Mock VLM：根据 prompt 变体模拟不同召回/精度行为。"""

    def __init__(self) -> None:
        self.rng = random.Random(42)

    def confirm(self, prompt: str, image_b64: str) -> dict[str, Any] | None:
        # 通过 prompt 关键字识别变体，模拟其行为分布
        if "confidence < 0.8 的项一律不输出" in prompt:
            # strict：高阈值，低召回低误报
            tpr, fpr = 0.5, 0.1
        elif "confidence >= 0.3 即可输出" in prompt:
            # loose：低阈值，高召回高误报
            tpr, fpr = 0.9, 0.5
        else:
            # baseline：平衡
            tpr, fpr = 0.75, 0.25
        # mock 模式下解码图像，检测是否含注入的亮块（与 GT 对齐）
        # 这样 TPR/FPR 才能反映 prompt 变体的差异，而非纯随机
        try:
            import base64 as _b64
            raw = _b64.b64decode(image_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                # 注入的劣化区域是 40x40 亮块（均值 > 200）
                bright_ratio = float((img.mean(axis=2) > 180).mean())
                gt_likely_degraded = bright_ratio > 0.05
            else:
                gt_likely_degraded = self.rng.random() < 0.5
        except Exception:
            gt_likely_degraded = self.rng.random() < 0.5
        if gt_likely_degraded:
            predicted = self.rng.random() < tpr
        else:
            predicted = self.rng.random() < fpr
        if not predicted:
            return {"findings": [], "overall_assessment": "无明显异常"}
        return {
            "findings": [
                {
                    "degradation_type": "mock_artifact",
                    "region_description": "mock 区域",
                    "severity": "moderate",
                    "confidence": 0.7,
                    "reasoning": "mock",
                    "mos_impact_estimate": -0.2,
                }
            ],
            "overall_assessment": "存在异常",
        }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="D2 VLM prompt 消融实验")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="基准集 manifest.json 路径")
    parser.add_argument("--samples", type=int, default=None,
                        help="只跑前 N 个样本（调试用）")
    parser.add_argument("--mock", action="store_true",
                        help="使用 mock VLM（无需 ollama），生成示例结果表")
    parser.add_argument("--out", type=Path, default=None,
                        help="将 markdown 表写入文件")
    args = parser.parse_args()

    # 加载样本
    if args.mock:
        n = args.samples or 20
        samples = generate_mock_samples(n)
        print(f"[ablation] mock 模式：生成 {len(samples)} 个合成样本")
    else:
        try:
            samples = load_samples(args.manifest, args.samples)
        except FileNotFoundError as exc:
            print(f"[ablation] 错误：{exc}", file=sys.stderr)
            print("[ablation] 提示：先运行 scripts/generate_benchmark_dataset.py 生成基准集，"
                  "或加 --mock 跑流程演示。", file=sys.stderr)
            return 1
        if not samples:
            print("[ablation] 错误：manifest 为空", file=sys.stderr)
            return 1
        print(f"[ablation] 加载 {len(samples)} 个样本 from {args.manifest}")

    # 构建 VLM client
    if args.mock:
        vlm: VLMClient = _MockAblationVLM()
    else:
        cfg = load_config()
        vlm = build_vlm_client(cfg.vlm)

    # 跑三种变体
    results: list[VariantResult] = []
    for variant in ("baseline", "strict", "loose"):
        print(f"[ablation] 跑变体 {variant} ...")
        r = evaluate_variant(variant, vlm, samples)
        results.append(r)
        print(f"  TPR={r.tpr:.3f} FPR={r.fpr:.3f} avg_findings={r.avg_findings:.2f} "
              f"avg_lat={r.avg_latency_ms:.1f}ms")

    # 输出 markdown 表
    table = render_markdown_table(results)
    print("\n" + table)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
        print(f"\n[ablation] 表已写入 {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
