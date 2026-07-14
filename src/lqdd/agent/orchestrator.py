from __future__ import annotations

import time
from typing import Any

from lqdd.agent.context import create_context
from lqdd.agent.judge_client import (
    JudgeClient,
    MockJudgeClient,
    OllamaJudgeClient,
    RuleBasedJudgeClient,
    build_agent_observe_prompt,
    parse_agent_step,
    run_judge,
)
from lqdd.agent.router import route_nominations
from lqdd.config.loader import AppConfig
from lqdd.detectors.registry import build_detector_registry, run_detectors
from lqdd.global_scan.scanner import GlobalScanner
from lqdd.models.agent import AgentAction, AgentContext, AgentMeta, AgentStep
from lqdd.models.inputs import SingleFrameInput
from lqdd.models.report import DegradationItem, PerformanceMetrics, QualityReport, TraceEntry
from lqdd.report.generator import ReportGenerator
from lqdd.vlm.client import VLMClient
from lqdd.vlm.confirm import build_vlm_client, run_vlm_confirm_for_item
from lqdd.vlm.fuse import fuse_all


def build_judge_client_from_config(config: AppConfig, mock: Any = None) -> JudgeClient:
    if mock is not None:
        return mock
    if config.judge.provider == "ollama":
        return OllamaJudgeClient(config.judge)
    return RuleBasedJudgeClient(config.agent)


# ---------------------------------------------------------------------------
# ReAct Agent 工具执行器
# ---------------------------------------------------------------------------


def _execute_vlm_analyze(
    ctx: AgentContext,
    action: AgentAction,
    vlm_client: VLMClient,
    config: AppConfig,
) -> str:
    """执行 vlm_analyze 工具：对指定 degradation_id 进行 VLM 视觉确认。
    如果未指定 degradation_id，则对所有未经 VLM 确认的检测项进行确认。"""
    if ctx.frame_input is None:
        return "vlm_analyze 失败：frame_input 不可用"

    targets = ctx.merged_degradations
    if action.degradation_id:
        targets = [d for d in ctx.merged_degradations if d.degradation_id == action.degradation_id]
        if not targets:
            return f"vlm_analyze 失败：未找到 degradation_id={action.degradation_id}"

    # 只对尚未经过 VLM 确认的检测项进行确认
    targets = [d for d in targets if d.vlm_reasoning is None]
    if not targets:
        return "vlm_analyze 跳过：目标检测项已有 VLM 结论"

    confirmed = 0
    for deg in targets:
        if ctx.vlm_calls_count >= config.vlm.max_calls_per_frame:
            break
        result = run_vlm_confirm_for_item(deg, ctx, vlm_client, config.agent, config.vlm)
        if result is not None:
            ctx.vlm_results.append(result)
            confirmed += 1

    # 重新融合
    ctx.merged_degradations = fuse_all(ctx.merged_degradations, ctx.vlm_results)
    return f"vlm_analyze 完成：确认了 {confirmed} 个检测项，当前 VLM 调用总计 {ctx.vlm_calls_count} 次"


def _execute_rerun_detector(
    ctx: AgentContext,
    action: AgentAction,
    config: AppConfig,
    registry: dict,
) -> str:
    """执行 rerun_detector 工具：用调整后的阈值重新运行指定检测器。"""
    if not action.detector:
        return "rerun_detector 失败：未指定 detector"
    if ctx.frame_input is None or ctx.scan_output is None:
        return "rerun_detector 失败：frame_input 或 scan_output 不可用"

    det = registry.get(action.detector)
    if det is None:
        return f"rerun_detector 失败：未知检测器 {action.detector}"

    # 应用阈值 delta（如果检测器支持）
    if action.nomination_threshold_delta is not None:
        orig = getattr(det, "config", None)
        if orig and hasattr(orig, "nomination_threshold"):
            orig.nomination_threshold += action.nomination_threshold_delta

    new_results = det.detect(ctx.frame_input, ctx.scan_output)
    # 去重：不重复添加已有同类型检测项
    existing_types = {d.degradation_type for d in ctx.merged_degradations}
    added = [r for r in new_results if r.degradation_type not in existing_types]
    ctx.merged_degradations.extend(added)
    return f"rerun_detector {action.detector} 完成：新增 {len(added)} 个检测项"


def _execute_dispatch_compression(
    ctx: AgentContext,
    config: AppConfig,
    registry: dict,
) -> str:
    """执行 dispatch_compression 工具：补充运行压缩伪影检测器。"""
    if any(d.detector == "compression_artifact" for d in ctx.merged_degradations):
        return "dispatch_compression 跳过：压缩伪影已检出"
    if ctx.frame_input is None or ctx.scan_output is None:
        return "dispatch_compression 失败：frame_input 或 scan_output 不可用"

    det = registry.get("compression_artifact")
    if det is None:
        return "dispatch_compression 失败：compression_artifact 检测器不可用"

    new_results = det.detect(ctx.frame_input, ctx.scan_output)
    ctx.merged_degradations.extend(new_results)
    return f"dispatch_compression 完成：新增 {len(new_results)} 个压缩伪影检测项"


# ---------------------------------------------------------------------------
# ReAct Agent 主循环
# ---------------------------------------------------------------------------


def run_react_agent(
    ctx: AgentContext,
    judge_client: JudgeClient,
    vlm_client: VLMClient,
    config: AppConfig,
    registry: dict,
    skipped_detectors: list[str],
    traces: list[TraceEntry],
) -> list[DegradationItem]:
    """ReAct Agent 核心循环。

    流程：
      1. LLM 观察当前 CV 检测结果（Observe）
      2. LLM 思考并自主选择下一步行动（Thought + Action）
      3. 执行工具，获取结果（Act + Observe）
      4. 重复直到 LLM 输出 accept 或达到最大步数
    """
    max_steps = config.agent.max_rounds * 3  # 给 Agent 充足的决策步数
    step = 1
    history: list[AgentStep] = []
    total_judge_ms = 0.0

    while step <= max_steps:
        # --- Observe: 构建当前状态给 LLM ---
        system_prompt, user_prompt = build_agent_observe_prompt(
            ctx.merged_degradations,
            config.report,
            skipped_detectors,
            step,
            max_steps,
            history,
        )

        # --- Think + Act: LLM 自主决策 ---
        t0 = time.perf_counter()
        raw = judge_client.decide(system_prompt, user_prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        total_judge_ms += latency_ms

        if raw is None:
            # LLM 不可用，降级到规则
            fallback = RuleBasedJudgeClient(config.agent)
            raw = fallback.decide(system_prompt, user_prompt)

        if raw is None:
            # 完全失败，直接接受
            raw = {"thought": "Agent 不可用，接受当前结果", "action": "accept", "reason": "服务不可用"}

        agent_step = parse_agent_step(raw, step, latency_ms)
        if agent_step is None:
            # 无法解析，强制 accept
            agent_step = AgentStep(
                step_index=step,
                thought=str(raw.get("thought", "解析失败")),
                action=AgentAction(action="accept", reason="动作解析失败，强制终止"),
                latency_ms=latency_ms,
            )

        action = agent_step.action

        # --- 防护：检测重复行动，避免小模型陷入循环 ---
        if action.action != "accept" and len(history) >= 1:
            recent_actions = [s.action.action for s in history[-2:]]
            if recent_actions.count(action.action) >= 2:
                # 同一个 action 已连续出现 2 次，强制终止
                agent_step = AgentStep(
                    step_index=step,
                    thought=f"检测到重复行动 {action.action}，强制终止循环",
                    action=AgentAction(action="accept", reason=f"防止循环：{action.action} 已重复执行"),
                    latency_ms=latency_ms,
                )
                action = agent_step.action

        # --- Execute Tool: 执行工具 ---
        observation = ""
        if action.action == "accept":
            observation = f"Agent 终止：{action.reason}"
            agent_step.observation = observation
            history.append(agent_step)
            ctx.agent_steps.append(agent_step)

            traces.append(
                TraceEntry(
                    stage="agent_step",
                    module="ReactAgent",
                    timestamp_ms=0.0,
                    duration_ms=latency_ms,
                    input_summary={"step": step, "thought": agent_step.thought},
                    output_summary={"action": "accept", "observation": observation},
                    decision=f"agent_accept_step{step}",
                    mode="fast",
                )
            )
            break

        elif action.action == "vlm_analyze":
            observation = _execute_vlm_analyze(ctx, action, vlm_client, config)

        elif action.action == "rerun_detector":
            observation = _execute_rerun_detector(ctx, action, config, registry)

        elif action.action == "dispatch_compression":
            observation = _execute_dispatch_compression(ctx, config, registry)

        agent_step.observation = observation
        history.append(agent_step)
        ctx.agent_steps.append(agent_step)

        traces.append(
            TraceEntry(
                stage="agent_step",
                module="ReactAgent",
                timestamp_ms=0.0,
                duration_ms=latency_ms,
                input_summary={"step": step, "thought": agent_step.thought},
                output_summary={"action": action.action, "observation": observation},
                decision=f"agent_{action.action}_step{step}",
                mode="fast",
            )
        )
        step += 1

    ctx.judge_ms = total_judge_ms
    return ctx.merged_degradations


# ---------------------------------------------------------------------------
# AgentOrchestrator
# ---------------------------------------------------------------------------


class AgentOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        vlm_client: VLMClient | None = None,
        judge_client: Any = None,
    ) -> None:
        self.config = config
        self.scanner = GlobalScanner(config.global_scan)
        self.registry = build_detector_registry(config)
        self.reporter = ReportGenerator(config.report)
        self._vlm_client = vlm_client
        self._judge_client = judge_client

    def run(self, frame_input: SingleFrameInput) -> QualityReport:
        t0 = time.perf_counter()
        ctx = create_context(frame_input, self.scanner.scan(frame_input), self.config.agent.max_rounds)
        ctx.frame_input = frame_input
        traces: list[TraceEntry] = []

        # --- Stage 1: mode_select ---
        traces.append(
            TraceEntry(
                stage="mode_select",
                module="AgentOrchestrator",
                timestamp_ms=0.0,
                duration_ms=0.0,
                input_summary={"requested_mode": frame_input.mode},
                output_summary={"selected_mode": "fast", "agent": True, "react_agent": True},
                decision="mode_fast_react_agent",
                mode="fast",
            )
        )

        # --- Stage 2: global_scan ---
        scan = ctx.scan_output
        assert scan is not None
        traces.append(
            TraceEntry(
                stage="global_scan",
                module="GlobalScanner",
                timestamp_ms=0.0,
                duration_ms=scan.scan_duration_ms,
                input_summary={"frame_id": frame_input.frame_id},
                output_summary={"nominations": len(scan.nominations)},
                decision="scan_complete",
                mode="fast",
            )
        )

        # --- Stage 3: routing（只做检测器派发，不再硬编码 VLM 路由）---
        dispatched = route_nominations(scan, self.config.agent, ctx)
        traces.append(
            TraceEntry(
                stage="routing",
                module="FastRouter",
                timestamp_ms=scan.scan_duration_ms,
                duration_ms=0.0,
                input_summary={"nominations": len(scan.nominations)},
                output_summary={
                    "dispatched": dispatched,
                    "decisions": [d.decision for d in ctx.routing_decisions],
                },
                decision="route_complete",
                mode="fast",
            )
        )

        # --- Stage 4: CV 检测 ---
        t_det = time.perf_counter()
        degradations = run_detectors(self.registry, dispatched, frame_input, scan)
        det_ms = (time.perf_counter() - t_det) * 1000.0
        ctx.preliminary_degradations = degradations
        ctx.merged_degradations = list(degradations)  # 初始化 merged 为 CV 结果

        traces.append(
            TraceEntry(
                stage="detection",
                module="SubDetectors",
                timestamp_ms=scan.scan_duration_ms,
                duration_ms=det_ms,
                input_summary={"detectors": dispatched},
                output_summary={"count": len(degradations)},
                decision="detectors_complete",
                mode="fast",
            )
        )

        # --- Stage 5: ReAct Agent 循环（LLM 自主决策是否调 VLM / 补检）---
        vlm_client = build_vlm_client(self.config.vlm, self._vlm_client)
        judge_client = build_judge_client_from_config(self.config, self._judge_client)

        skipped = [d for d in ("edge_bleed", "compression_artifact") if d not in dispatched]

        ctx.merged_degradations = run_react_agent(
            ctx=ctx,
            judge_client=judge_client,
            vlm_client=vlm_client,
            config=self.config,
            registry=self.registry,
            skipped_detectors=skipped,
            traces=traces,
        )

        # --- Stage 6: aggregation ---
        t_agg = time.perf_counter()
        total_ms = (time.perf_counter() - t0) * 1000.0
        agg_ms = (time.perf_counter() - t_agg) * 1000.0

        perf = PerformanceMetrics(
            total_ms=total_ms,
            global_scan_ms=scan.scan_duration_ms,
            detection_ms=det_ms,
            aggregation_ms=agg_ms,
            vlm_ms=ctx.vlm_ms,
            judge_ms=ctx.judge_ms,
        )

        traces.append(
            TraceEntry(
                stage="aggregation",
                module="ReportGenerator",
                timestamp_ms=0.0,
                duration_ms=agg_ms,
                input_summary={"degradation_count": len(ctx.merged_degradations)},
                output_summary={"agent_steps": len(ctx.agent_steps)},
                decision="aggregate_mos",
                mode="fast",
            )
        )

        # 判断是否由 Agent 自主触发了 VLM
        agent_driven_vlm = any(s.action.action == "vlm_analyze" for s in ctx.agent_steps)

        agent_meta = AgentMeta(
            rounds_executed=ctx.round_index,
            max_rounds_reached=ctx.max_rounds_reached,
            vlm_calls=ctx.vlm_calls_count,
            judge_assessment=None,  # ReAct 模式下没有 judge assessment 概念
            agent_steps=[
                {
                    "step": s.step_index,
                    "thought": s.thought,
                    "action": s.action.action,
                    "reason": s.action.reason,
                    "observation": s.observation,
                    "latency_ms": round(s.latency_ms, 1),
                }
                for s in ctx.agent_steps
            ],
            agent_driven_vlm=agent_driven_vlm,
        )

        vlm_summary = [
            {
                "degradation_id": d.degradation_id,
                "reasoning": d.vlm_reasoning["reasoning"],
                "region_type": d.region_type,
            }
            for d in ctx.merged_degradations
            if d.vlm_reasoning
        ]

        return self.reporter.generate(
            frame_input,
            scan,
            ctx.merged_degradations,
            traces,
            perf,
            agent_meta=agent_meta,
            vlm_summary=vlm_summary or None,
            vlm_ms=ctx.vlm_ms,
            judge_ms=ctx.judge_ms,
        )
