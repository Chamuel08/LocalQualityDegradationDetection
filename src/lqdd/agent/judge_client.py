from __future__ import annotations

import base64
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from lqdd.agent.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE
from lqdd.config.loader import AgentConfig, JudgeConfig
from lqdd.models.agent import AgentAction, AgentContext, JudgeOutput
from lqdd.models.report import DegradationItem, TraceEntry
from lqdd.report.generator import compute_mos


WHITELIST = frozenset({"vlm_analyze", "rerun_detector", "dispatch_compression", "accept"})


class JudgeClient(ABC):
    @abstractmethod
    def review(self, prompt: str) -> dict[str, Any] | None:
        ...


class OllamaJudgeClient(JudgeClient):
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    def review(self, prompt: str) -> dict[str, Any] | None:
        url = f"{self.config.host.rstrip('/')}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
        }
        try:
            with httpx.Client(timeout=self.config.timeout_ms / 1000.0, trust_env=False) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                return json.loads(content)
        except Exception:
            return None


class MockJudgeClient(JudgeClient):
    def __init__(self, fixture_path: Path | None = None) -> None:
        self.fixture_path = fixture_path

    def review(self, prompt: str) -> dict[str, Any] | None:
        if self.fixture_path and self.fixture_path.is_file():
            return json.loads(self.fixture_path.read_text(encoding="utf-8"))
        return {
            "assessment": "consistent",
            "reasoning": "mock judge：结果一致",
            "actions": [],
            "needs_round2": False,
        }


class RuleBasedJudgeClient(JudgeClient):
    """Fallback when LLM unavailable — handles MOS-low-no-detection."""

    def __init__(self, agent_cfg: AgentConfig) -> None:
        self.agent_cfg = agent_cfg

    def review(self, prompt: str) -> dict[str, Any] | None:
        if '"detection_count": 0' in prompt or "detections: []" in prompt:
            if "global_mos" in prompt:
                try:
                    import re

                    m = re.search(r"global_mos[\":\s]+([0-9.]+)", prompt)
                    mos = float(m.group(1)) if m else 4.5
                except (ValueError, AttributeError):
                    mos = 4.5
                if mos < 4.0:
                    return {
                        "assessment": "inconsistent",
                        "reasoning": "全局 MOS 偏低但未检出劣化，建议补检压缩伪影",
                        "actions": [{"action": "dispatch_compression", "reason": "MOS 低无检出"}],
                        "needs_round2": True,
                    }
        return {
            "assessment": "consistent",
            "reasoning": "规则审查：未发现明显矛盾",
            "actions": [],
            "needs_round2": False,
        }


def _parse_action(raw: dict[str, Any]) -> AgentAction | None:
    action = raw.get("action")
    if action not in WHITELIST:
        return None
    delta = raw.get("nomination_threshold_delta")
    if action == "rerun_detector" and delta is not None:
        delta_f = float(delta)
        if not (-0.15 <= delta_f <= -0.05):
            return None
    return AgentAction(
        action=action,
        target_region=raw.get("target_region"),
        detector=raw.get("detector"),
        nomination_threshold_delta=float(delta) if delta is not None else None,
        reason=raw.get("reason"),
    )


def parse_judge_response(raw: dict[str, Any], latency_ms: float = 0.0) -> JudgeOutput:
    rejected: list[str] = []
    actions: list[AgentAction] = []
    for item in raw.get("actions", []):
        if not isinstance(item, dict):
            continue
        parsed = _parse_action(item)
        if parsed:
            actions.append(parsed)
        else:
            rejected.append(str(item.get("action", item)))
    assessment = raw.get("assessment", "uncertain")
    if assessment not in ("consistent", "uncertain", "inconsistent"):
        assessment = "uncertain"
    return JudgeOutput(
        assessment=assessment,
        reasoning=str(raw.get("reasoning", "")),
        actions=actions,
        needs_round2=bool(raw.get("needs_round2", False)),
        judge_latency_ms=latency_ms,
        raw_rejected_actions=rejected,
    )


def build_judge_prompt(
    degradations: list[DegradationItem],
    report_cfg,
    skipped: list[str],
    vlm_calls: int,
) -> str:
    mos, _ = compute_mos(degradations, report_cfg)
    detections = [
        {
            "detector": d.detector,
            "confidence": d.confidence,
            "degradation_type": d.degradation_type,
            "severity": d.severity,
        }
        for d in degradations
    ]
    return JUDGE_USER_TEMPLATE.format(
        global_mos=mos,
        detection_count=len(degradations),
        detections_json=json.dumps(detections, ensure_ascii=False),
        skipped=json.dumps(skipped, ensure_ascii=False),
        vlm_calls=vlm_calls,
    )


def run_judge(
    ctx: AgentContext,
    client: JudgeClient,
    report_cfg,
    skipped_detectors: list[str] | None = None,
) -> JudgeOutput:
    t0 = time.perf_counter()
    prompt = build_judge_prompt(
        ctx.merged_degradations,
        report_cfg,
        skipped_detectors or [],
        ctx.vlm_calls_count,
    )
    raw = client.review(prompt)
    latency = (time.perf_counter() - t0) * 1000.0
    if raw is None:
        fallback = RuleBasedJudgeClient(AgentConfig())
        raw = fallback.review(prompt)
    if raw is None:
        return JudgeOutput(
            assessment="uncertain",
            reasoning="Judge 不可用，跳过 Round 2",
            actions=[],
            needs_round2=False,
            judge_latency_ms=latency,
        )
    output = parse_judge_response(raw, latency)
    if output.raw_rejected_actions:
        ctx.traces.append(
            TraceEntry(
                stage="judge",
                module="LLMJudge",
                timestamp_ms=0.0,
                duration_ms=latency,
                input_summary={"rejected_actions": output.raw_rejected_actions},
                output_summary={},
                decision="judge_action_rejected",
                mode="fast",
            )
        )
    return output
