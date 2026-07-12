from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from lqdd.models.enums import RootCauseCategory, Severity


@dataclass
class Evidence:
    method: str
    metric: str
    value: float
    threshold: float
    detail: str
    attention_map: str | None = None


@dataclass
class RootCauseHypothesis:
    cause: str
    confidence: float


@dataclass
class DegradationItem:
    degradation_id: str
    region_type: str
    degradation_type: str
    severity: str
    confidence: float
    mos_impact: float
    bbox: list[int]
    frame_indices: list[int]
    description: str
    detector: str
    evidence: Evidence
    root_cause_hypothesis: RootCauseHypothesis
    region_mask_rle: str | None = None
    vlm_reasoning: dict[str, Any] | None = None


@dataclass
class PenaltyItem:
    source: str
    penalty: float
    effective_penalty: float
    decay_index: int
    reason: str


@dataclass
class MOSBreakdown:
    base_mos: float
    total_penalty: float
    cap_applied: bool
    cap_reason: str | None
    penalties: list[PenaltyItem]


@dataclass
class TraceEntry:
    stage: str
    module: str
    timestamp_ms: float
    duration_ms: float
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    decision: str
    mode: Literal["fast", "deep"] = "fast"


@dataclass
class PerformanceMetrics:
    total_ms: float
    global_scan_ms: float
    detection_ms: float
    aggregation_ms: float
    vlm_ms: float = 0.0
    judge_ms: float = 0.0


@dataclass
class DegradationSummary:
    total_count: int
    by_severity: dict[str, int]
    by_detector: dict[str, int]
    by_root_cause: dict[str, int] = field(default_factory=dict)
    top_issues: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    report_id: str
    video_id: str
    mode: Literal["fast", "deep"]
    frame_index: int
    report_timestamp: str
    system_version: str
    overall_mos: float
    severity: str
    degradations: list[DegradationItem]
    decision_trace: list[TraceEntry]
    performance: PerformanceMetrics
    mos_breakdown: MOSBreakdown | None = None
    degradation_summary: DegradationSummary | None = None
    vlm_reasoning_summary: list | None = None
    agent_meta: dict[str, Any] | None = None


def evidence_to_dict(e: Evidence) -> dict[str, Any]:
    d = asdict(e)
    return d


def degradation_to_dict(d: DegradationItem) -> dict[str, Any]:
    return {
        "degradation_id": d.degradation_id,
        "region_type": d.region_type,
        "degradation_type": d.degradation_type,
        "severity": d.severity,
        "confidence": d.confidence,
        "mos_impact": d.mos_impact,
        "bbox": d.bbox,
        "region_mask_rle": d.region_mask_rle,
        "frame_indices": d.frame_indices,
        "description": d.description,
        "detector": d.detector,
        "evidence": evidence_to_dict(d.evidence),
        "root_cause_hypothesis": asdict(d.root_cause_hypothesis),
        "vlm_reasoning": d.vlm_reasoning,
    }


def trace_to_dict(t: TraceEntry) -> dict[str, Any]:
    return asdict(t)


def report_to_dict(report: QualityReport) -> dict[str, Any]:
    out: dict[str, Any] = {
        "report_id": report.report_id,
        "video_id": report.video_id,
        "mode": report.mode,
        "frame_index": report.frame_index,
        "report_timestamp": report.report_timestamp,
        "system_version": report.system_version,
        "overall_mos": round(report.overall_mos, 3),
        "severity": report.severity,
        "degradations": [degradation_to_dict(d) for d in report.degradations],
        "decision_trace": [trace_to_dict(t) for t in report.decision_trace],
        "performance": asdict(report.performance),
        "vlm_reasoning_summary": report.vlm_reasoning_summary,
    }
    if report.mos_breakdown:
        out["mos_breakdown"] = asdict(report.mos_breakdown)
    if report.degradation_summary:
        out["degradation_summary"] = asdict(report.degradation_summary)
    if report.agent_meta:
        out["agent_meta"] = report.agent_meta
    return out
