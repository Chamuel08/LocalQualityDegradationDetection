from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from lqdd.config.loader import ReportConfig
from lqdd.models.enums import Severity
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput
from lqdd.models.report import (
    DegradationItem,
    DegradationSummary,
    MOSBreakdown,
    PerformanceMetrics,
    QualityReport,
    TraceEntry,
)
from lqdd.attribution.scenario import attribute_scenarios, scenario_attribution_to_dict


SEVERITY_ORDER = {
    Severity.GOOD.value: 0,
    Severity.MINOR.value: 1,
    Severity.MODERATE.value: 2,
    Severity.SEVERE.value: 3,
    Severity.CRITICAL.value: 4,
}


def _overall_severity(degradations: list[DegradationItem]) -> str:
    if not degradations:
        return Severity.GOOD.value
    return max(degradations, key=lambda d: SEVERITY_ORDER.get(d.severity, 0)).severity


def compute_mos(
    degradations: list[DegradationItem],
    cfg: ReportConfig,
    frame_bgr: "np.ndarray | None" = None,
) -> tuple[float | None, MOSBreakdown]:
    """计算帧级 MOS 分数（单一总分）。

    **MOS 与归因解耦**：归因（劣化是什么 / 在哪 / 为什么）来自 ``degradations[]``
    （detector / bbox / evidence / root_cause / vlm_reasoning），与本函数无关。
    本函数只产出帧级一个总分，由 ``cfg.mos_model`` 指定的无参考画质模型直接预测，
    **不再硬编码 per-distortion 扣分、不再有 base_mos/decay 衰减公式**。

    后端（由 ``cfg.mos_model`` 控制）：

    1. ``"clip_iqa"``（默认，需 pyiqa + torch）
       CLIP-IQA（ICCV 2023）无参考画质预测，[0,1] 线性映射到 MOS [1,5]。
       需传入 ``frame_bgr``（BGR uint8）。
       **失败语义**：依赖缺失 / 权重下载失败 / 推理异常 / 未提供帧时，
       返回 ``(None, MOSBreakdown(status="unavailable", reason=...))``，
       由上层把 ``overall_mos`` 置 null 并记录原因，**绝不回退到硬编码默认分**。

    2. ``"internal"``（预留，需自行实现 ``lqdd.mos.internal_model``）
       未实现时同样返回 unavailable，不回退。

    Args:
        degradations: 当前帧所有检出的劣化项（归因来源，与 MOS 计算独立）
        cfg:          ReportConfig（mos_model / system_version）
        frame_bgr:    原始帧图像（BGR uint8），clip_iqa 模式下必须提供

    Returns:
        (mos_score | None, MOSBreakdown) 元组。mos_score 为 None 表示 MOS 不可用。
    """
    model = cfg.mos_model or "clip_iqa"

    # ------------------------------------------------------------------ #
    # internal 分支：预留接口（未实现 → unavailable，不回退）              #
    # ------------------------------------------------------------------ #
    if model == "internal":
        try:
            from lqdd.mos.internal_model import predict_mos  # type: ignore[import-not-found]

            return predict_mos(degradations, cfg, frame_bgr)
        except ImportError as exc:
            reason = f"internal MOS 后端未实现（{exc}）；需实现 lqdd.mos.internal_model"
            return None, MOSBreakdown(model="internal", mos=None, status="unavailable", reason=reason)

    # ------------------------------------------------------------------ #
    # clip_iqa 分支（默认）：CLIP-IQA 直接预测帧级 MOS                     #
    # ------------------------------------------------------------------ #
    if frame_bgr is None:
        reason = "clip_iqa 需要原始帧 frame_bgr，未提供"
        return None, MOSBreakdown(model="clip_iqa", mos=None, status="unavailable", reason=reason)

    try:
        from lqdd.mos.clip_iqa import predict_mos_clip_iqa

        mos = predict_mos_clip_iqa(frame_bgr)
        return mos, MOSBreakdown(model="clip_iqa", mos=mos, status="ok", reason=None)
    except ImportError as exc:
        reason = (
            f"CLIP-IQA 不可用（{exc}）；安装方法：pip install \"lqdd[clip_iqa]\""
            "（即 pyiqa + torch + torchvision）。未安装时 overall_mos 置 null，不回退默认分。"
        )
        return None, MOSBreakdown(model="clip_iqa", mos=None, status="unavailable", reason=reason)
    except Exception as exc:
        reason = f"CLIP-IQA 推理失败（{exc}）；overall_mos 置 null，不回退默认分"
        return None, MOSBreakdown(model="clip_iqa", mos=None, status="unavailable", reason=reason)


def build_summary(degradations: list[DegradationItem]) -> DegradationSummary:
    by_severity: dict[str, int] = {}
    by_detector: dict[str, int] = {}
    by_root: dict[str, int] = {}
    for d in degradations:
        by_severity[d.severity] = by_severity.get(d.severity, 0) + 1
        by_detector[d.detector] = by_detector.get(d.detector, 0) + 1
        by_root[d.root_cause_hypothesis.cause] = by_root.get(d.root_cause_hypothesis.cause, 0) + 1
    top = [d.description for d in sorted(degradations, key=lambda x: x.confidence, reverse=True)[:3]]
    return DegradationSummary(
        total_count=len(degradations),
        by_severity=by_severity,
        by_detector=by_detector,
        by_root_cause=by_root,
        top_issues=top,
    )


class ReportGenerator:
    def __init__(self, config: ReportConfig) -> None:
        self.config = config

    def generate(
        self,
        frame_input: SingleFrameInput,
        scan_output: GlobalScanOutput,
        degradations: list[DegradationItem],
        traces: list[TraceEntry],
        perf: PerformanceMetrics,
        agent_meta: Any | None = None,
        vlm_summary: list | None = None,
        vlm_ms: float = 0.0,
        judge_ms: float = 0.0,
        quality_caption: dict[str, Any] | None = None,
    ) -> QualityReport:
        mos, breakdown = compute_mos(degradations, self.config, frame_bgr=frame_input.frame)
        mos_unavailable_reason = breakdown.reason if (mos is None and breakdown is not None) else None
        perf = PerformanceMetrics(
            total_ms=perf.total_ms,
            global_scan_ms=perf.global_scan_ms,
            detection_ms=perf.detection_ms,
            aggregation_ms=perf.aggregation_ms,
            vlm_ms=vlm_ms,
            judge_ms=judge_ms,
        )
        # 业务场景归因：劣化 → 业务场景 → 修复建议
        scenario_attribution = [
            scenario_attribution_to_dict(a) for a in attribute_scenarios(degradations)
        ] or None
        return QualityReport(
            report_id=f"rpt_{uuid.uuid4().hex[:12]}",
            video_id=frame_input.frame_id,
            mode="fast",
            frame_index=scan_output.frame_index,
            report_timestamp=datetime.now(timezone.utc).isoformat(),
            system_version=self.config.system_version,
            overall_mos=mos,
            severity=_overall_severity(degradations),
            degradations=degradations,
            decision_trace=traces,
            performance=perf,
            mos_breakdown=breakdown,
            degradation_summary=build_summary(degradations),
            vlm_reasoning_summary=vlm_summary,
            agent_meta=asdict(agent_meta) if hasattr(agent_meta, "__dataclass_fields__") else agent_meta,
            scenario_attribution=scenario_attribution,
            quality_caption=quality_caption,
            mos_unavailable_reason=mos_unavailable_reason,
        )
