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
) -> tuple[float, MOSBreakdown]:
    """计算帧级 MOS 分数（单一总分）。

    **MOS 与归因解耦**：归因（劣化是什么 / 在哪 / 为什么）来自 `degradations[]`
    （detector / bbox / evidence / root_cause / vlm_reasoning），与本函数无关。
    本函数只产出帧级一个总分；`mos_breakdown.penalties` 在 `clip_iqa` 模式下为空，
    在 `rule` 模式下为求和明细（非感知归因）。

    支持三种后端（由 cfg.mos_model 控制）：

    1. "rule"（默认，零额外依赖）
       MOS = base_mos + Σ( penalty_i × decay_factor^i )，工程启发，非主观拟合。

    2. "clip_iqa"（推荐，需 pyiqa + torch）
       CLIP-IQA（ICCV 2023）无参考画质预测，[0,1] 线性映射到 MOS [1,5]。
       需传入 frame_bgr（BGR uint8）。

    3. "internal"（预留，需自行实现 lqdd.mos.internal_model）

    Args:
        degradations: 当前帧所有检出的劣化项（归因来源，与 MOS 计算独立）
        cfg:          ReportConfig，包含 base_mos, decay_factor, mos_model
        frame_bgr:    原始帧图像（BGR uint8），clip_iqa 模式下必须提供

    Returns:
        (mos_score, MOSBreakdown) 元组
    """
    from lqdd.models.report import PenaltyItem

    # ------------------------------------------------------------------ #
    # clip_iqa 分支：CLIP-IQA 直接预测帧级 MOS                            #
    # ------------------------------------------------------------------ #
    if cfg.mos_model == "clip_iqa":
        if frame_bgr is None:
            # 没有图像帧时无法运行，降级到 rule
            import warnings
            warnings.warn(
                "mos_model='clip_iqa' 需要传入 frame_bgr 参数，当前降级到 rule 模式",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            try:
                from lqdd.mos.clip_iqa import predict_mos_clip_iqa

                mos = predict_mos_clip_iqa(frame_bgr)
                # CLIP-IQA 直接给出帧级 MOS，构造一个简洁的 breakdown 记录
                breakdown = MOSBreakdown(
                    base_mos=mos,
                    total_penalty=0.0,
                    cap_applied=False,
                    cap_reason="mos_model=clip_iqa，分数由 CLIP-IQA 直接预测",
                    penalties=[],
                )
                return mos, breakdown
            except ImportError as exc:
                import warnings
                warnings.warn(
                    f"CLIP-IQA 不可用（{exc}），降级到 rule 模式。"
                    "安装方法：pip install pyiqa torch torchvision setuptools",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"CLIP-IQA 推理失败（{exc}），降级到 rule 模式",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # ------------------------------------------------------------------ #
    # internal 分支：预留接口                                               #
    # ------------------------------------------------------------------ #
    elif cfg.mos_model == "internal":
        try:
            from lqdd.mos.internal_model import predict_mos  # type: ignore[import-not-found]
            return predict_mos(degradations, cfg, frame_bgr)
        except ImportError:
            import warnings
            warnings.warn(
                "mos_model='internal' 但 lqdd.mos.internal_model 未实现，降级到 rule",
                RuntimeWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------ #
    # rule 路径（默认 / 降级）                                              #
    # 按 mos_impact 绝对值降序，后续项以 decay_factor^index 衰减            #
    # 注意：per-item effective_penalty 是求和明细，非感知归因；归因看        #
    # degradations[]（detector/bbox/evidence/root_cause）。                #
    # ------------------------------------------------------------------ #
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
        mos, breakdown = compute_mos(degradations, self.config, frame_bgr=frame_input.frame)
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
