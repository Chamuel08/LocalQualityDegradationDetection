from __future__ import annotations

from copy import deepcopy

from lqdd.config.loader import AppConfig
from lqdd.detectors.compression.detector import CompressionArtifactDetector
from lqdd.detectors.edge_bleed.detector import EdgeBleedDetector
from lqdd.models.agent import AgentAction, AgentContext
from lqdd.models.report import DegradationItem


def execute_round2(
    ctx: AgentContext,
    config: AppConfig,
) -> list[DegradationItem]:
    if not ctx.judge_output or not ctx.judge_output.needs_round2:
        return ctx.merged_degradations

    degradations = list(ctx.merged_degradations)
    frame_input = ctx.frame_input
    scan = ctx.scan_output
    if frame_input is None or scan is None:
        return degradations

    edge = EdgeBleedDetector(config.edge_bleed)
    compression = CompressionArtifactDetector(config.compression)
    detectors = {"edge_bleed": edge, "compression_artifact": compression}

    for action in ctx.judge_output.actions:
        ctx.round2_actions_executed.append(action)
        if action.action == "accept":
            continue
        if action.action == "dispatch_compression":
            if not any(d.detector == "compression_artifact" for d in degradations):
                degradations.extend(compression.detect(frame_input, scan))
        elif action.action == "rerun_detector" and action.detector:
            det = detectors.get(action.detector)
            if det:
                degradations.extend(det.detect(frame_input, scan))
        elif action.action == "vlm_analyze":
            pass

    ctx.round_index = 2
    ctx.max_rounds_reached = True
    return degradations
