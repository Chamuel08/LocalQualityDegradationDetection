from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput
from lqdd.models.report import DegradationItem, TraceEntry


@dataclass
class RoutingDecision:
    source: Literal["nomination"] = "nomination"
    target_detector: str | None = None
    decision: Literal["dispatch", "skip", "vlm_pending"] = "dispatch"
    reason: str = ""
    confidence_band: Literal["high", "grey", "low"] | None = None
    region_type: str | None = None
    anomaly_score: float = 0.0


@dataclass
class VlmPendingItem:
    degradation_id: str
    detector: str
    bbox: tuple[int, int, int, int]
    preliminary_confidence: float
    region_type: str


@dataclass
class VLMResult:
    region_type: str
    degradation_type: str
    is_degraded: bool
    vlm_confidence: float
    vlm_severity: str
    reasoning: str
    mos_impact_estimate: float
    root_cause: str
    ux_impact: str
    vlm_latency_ms: float = 0.0


@dataclass
class AgentAction:
    action: Literal["vlm_analyze", "rerun_detector", "dispatch_compression", "accept"]
    target_region: str | None = None
    detector: str | None = None
    nomination_threshold_delta: float | None = None
    reason: str | None = None
    # ReAct Agent 新增：关联的检测项 ID（vlm_analyze 时使用）
    degradation_id: str | None = None


@dataclass
class AgentStep:
    """记录 ReAct Agent 一个完整的 Thought-Action-Observation 步骤。"""

    step_index: int
    thought: str
    action: AgentAction
    # Observation：工具执行后的结果描述
    observation: str = ""
    latency_ms: float = 0.0


@dataclass
class JudgeOutput:
    assessment: Literal["consistent", "uncertain", "inconsistent"]
    reasoning: str
    actions: list[AgentAction]
    needs_round2: bool
    judge_latency_ms: float = 0.0
    raw_rejected_actions: list[str] = field(default_factory=list)


@dataclass
class AgentMeta:
    rounds_executed: int = 1
    max_rounds_reached: bool = False
    vlm_calls: int = 0
    judge_assessment: str | None = None
    # ReAct Agent 新增：记录 Agent 自主决策步骤
    agent_steps: list[dict[str, Any]] = field(default_factory=list)
    # ReAct Agent 新增：是否由 Agent 自主决策触发 VLM（而非硬编码路由）
    agent_driven_vlm: bool = False


@dataclass
class AgentContext:
    round_index: int = 1
    max_rounds: int = 2
    mode: Literal["fast", "deep"] = "fast"
    frame_input: SingleFrameInput | None = None
    scan_output: GlobalScanOutput | None = None
    routing_decisions: list[RoutingDecision] = field(default_factory=list)
    preliminary_degradations: list[DegradationItem] = field(default_factory=list)
    merged_degradations: list[DegradationItem] = field(default_factory=list)
    pending_vlm: list[VlmPendingItem] = field(default_factory=list)
    vlm_results: list[VLMResult] = field(default_factory=list)
    judge_output: JudgeOutput | None = None
    round2_actions_executed: list[AgentAction] = field(default_factory=list)
    vlm_calls_count: int = 0
    max_rounds_reached: bool = False
    traces: list[TraceEntry] = field(default_factory=list)
    dispatched_detectors: list[str] = field(default_factory=list)
    vlm_ms: float = 0.0
    judge_ms: float = 0.0
    # ReAct Agent 新增：记录所有 Agent 自主决策步骤
    agent_steps: list[AgentStep] = field(default_factory=list)
