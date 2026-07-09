from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

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


def compute_mos(degradations: list[DegradationItem], cfg: ReportConfig) -> tuple[float, MOSBreakdown]:
    from lqdd.models.report import PenaltyItem

    sorted_items = sorted(degradations, key=lambda d: abs(d.mos_impact), reverse=True)
    penalties: list[PenaltyItem] = []
    total = 0.0
    for idx, item in enumerate(sorted_items):
        effective = item.mos_impact * (cfg.decay_factor**idx)
        penalties.append(
            PenaltyItem(
                source=item.degradation_id,
                penalty=item.mos_impact,
                effective_penalty=round(effective, 4),
                decay_index=idx,
                reason=item.description,
            )
        )
        total += effective
    mos = max(1.0, min(5.0, cfg.base_mos + total))
    breakdown = MOSBreakdown(
        base_mos=cfg.base_mos,
        total_penalty=round(total, 4),
        cap_applied=mos <= 1.0,
        cap_reason="overall_mos floored at 1.0" if mos <= 1.0 else None,
        penalties=penalties,
    )
    return round(mos, 3), breakdown


def build_summary(degradations: list[DegradationItem]) -> DegradationSummary:
    by_severity: dict[str, int] = {}
    by_detector: dict[str, int] = {}
    by_root: dict[str, int] = {}
    for d in degradations:
        by_severity[d.severity] = by_severity.get(d.severity, 0) + 1
        by_detector[d.detector] = by_detector.get(d.detector, 0) + 1
        by_root[d.root_cause_hypothesis.cause] = by_root.get(d.root_cause_hypothesis.cause, 0) + 1
    top = [d.description for d in sorted(degradations, key=lambda x: abs(x.mos_impact), reverse=True)[:3]]
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
    ) -> QualityReport:
        mos, breakdown = compute_mos(degradations, self.config)
        perf = PerformanceMetrics(
            total_ms=perf.total_ms,
            global_scan_ms=perf.global_scan_ms,
            detection_ms=perf.detection_ms,
            aggregation_ms=perf.aggregation_ms,
            vlm_ms=vlm_ms,
            judge_ms=judge_ms,
        )
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
        )
