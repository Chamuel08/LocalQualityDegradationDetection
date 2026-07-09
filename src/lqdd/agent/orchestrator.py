from __future__ import annotations

import time
from typing import Any

from lqdd.agent.actions import execute_round2
from lqdd.agent.context import create_context
from lqdd.agent.judge_client import MockJudgeClient, OllamaJudgeClient, RuleBasedJudgeClient, run_judge
from lqdd.agent.router import route_nominations
from lqdd.config.loader import AppConfig
from lqdd.detectors.compression.detector import CompressionArtifactDetector
from lqdd.detectors.edge_bleed.detector import EdgeBleedDetector
from lqdd.global_scan.scanner import GlobalScanner
from lqdd.models.agent import AgentMeta, AgentContext
from lqdd.models.inputs import SingleFrameInput
from lqdd.models.report import PerformanceMetrics, QualityReport, TraceEntry
from lqdd.report.generator import ReportGenerator
from lqdd.vlm.client import VLMClient
from lqdd.vlm.confirm import build_vlm_client, run_vlm_confirm
from lqdd.vlm.fuse import fuse_all


def build_judge_client_from_config(config: AppConfig, mock: Any = None):
    if mock is not None:
        return mock
    if config.judge.provider == "ollama":
        return OllamaJudgeClient(config.judge)
    return RuleBasedJudgeClient(config.agent)


class AgentOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        vlm_client: VLMClient | None = None,
        judge_client: Any = None,
    ) -> None:
        self.config = config
        self.scanner = GlobalScanner(config.global_scan)
        self.edge = EdgeBleedDetector(config.edge_bleed)
        self.compression = CompressionArtifactDetector(config.compression)
        self.reporter = ReportGenerator(config.report)
        self._vlm_client = vlm_client
        self._judge_client = judge_client

    def run(self, frame_input: SingleFrameInput) -> QualityReport:
        t0 = time.perf_counter()
        ctx = create_context(frame_input, self.scanner.scan(frame_input), self.config.agent.max_rounds)
        ctx.frame_input = frame_input
        traces: list[TraceEntry] = []

        traces.append(
            TraceEntry(
                stage="mode_select",
                module="AgentOrchestrator",
                timestamp_ms=0.0,
                duration_ms=0.0,
                input_summary={"requested_mode": frame_input.mode},
                output_summary={"selected_mode": "fast", "agent": True},
                decision="mode_fast_agent",
                mode="fast",
            )
        )

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

        t_det = time.perf_counter()
        degradations = []
        detectors = {
            "edge_bleed": self.edge,
            "compression_artifact": self.compression,
        }
        for name in dispatched:
            det = detectors.get(name)
            if det:
                degradations.extend(det.detect(frame_input, scan))
        det_ms = (time.perf_counter() - t_det) * 1000.0
        ctx.preliminary_degradations = degradations

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

        vlm_client = build_vlm_client(self.config.vlm, self._vlm_client)
        run_vlm_confirm(ctx, vlm_client, self.config.agent, self.config.vlm)
        traces.extend(ctx.traces)
        ctx.traces.clear()

        ctx.merged_degradations = fuse_all(ctx.preliminary_degradations, ctx.vlm_results)

        judge_client = build_judge_client_from_config(self.config, self._judge_client)
        skipped = [d for d in ("edge_bleed", "compression_artifact") if d not in dispatched]
        ctx.judge_output = run_judge(ctx, judge_client, self.config.report, skipped)

        traces.append(
            TraceEntry(
                stage="judge",
                module="LLMJudge",
                timestamp_ms=0.0,
                duration_ms=ctx.judge_output.judge_latency_ms,
                input_summary={"degradation_count": len(ctx.merged_degradations)},
                output_summary={
                    "assessment": ctx.judge_output.assessment,
                    "needs_round2": ctx.judge_output.needs_round2,
                    "actions": [a.action for a in ctx.judge_output.actions],
                },
                decision=f"judge_{ctx.judge_output.assessment}",
                mode="fast",
            )
        )
        ctx.judge_ms = ctx.judge_output.judge_latency_ms

        if ctx.judge_output.needs_round2 and ctx.round_index < ctx.max_rounds:
            ctx.merged_degradations = execute_round2(ctx, self.config)
            traces.append(
                TraceEntry(
                    stage="detection",
                    module="Round2Executor",
                    timestamp_ms=0.0,
                    duration_ms=0.0,
                    input_summary={"actions": [a.action for a in ctx.round2_actions_executed]},
                    output_summary={"count": len(ctx.merged_degradations)},
                    decision="round2_complete",
                    mode="fast",
                )
            )

        t_agg = time.perf_counter()
        total_ms = (time.perf_counter() - t0) * 1000.0
        agg_ms = (time.perf_counter() - t_agg) * 1000.0

        perf = PerformanceMetrics(
            total_ms=total_ms,
            global_scan_ms=scan.scan_duration_ms,
            detection_ms=det_ms,
            aggregation_ms=agg_ms,
        )

        traces.append(
            TraceEntry(
                stage="aggregation",
                module="ReportGenerator",
                timestamp_ms=0.0,
                duration_ms=agg_ms,
                input_summary={"degradation_count": len(ctx.merged_degradations)},
                output_summary={"rounds": ctx.round_index},
                decision="aggregate_mos",
                mode="fast",
            )
        )

        agent_meta = AgentMeta(
            rounds_executed=ctx.round_index,
            max_rounds_reached=ctx.max_rounds_reached,
            vlm_calls=ctx.vlm_calls_count,
            judge_assessment=ctx.judge_output.assessment if ctx.judge_output else None,
        )

        vlm_summary = [
            {"degradation_id": d.degradation_id, "reasoning": d.vlm_reasoning["reasoning"], "region_type": d.region_type}
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
