from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

from lqdd.agent.prompts import (
    AGENT_OBSERVE_TEMPLATE,
    AGENT_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
)
from lqdd.config.loader import AgentConfig, JudgeConfig
from lqdd.models.agent import AgentAction, AgentContext, AgentStep, JudgeOutput
from lqdd.models.report import DegradationItem, TraceEntry
from lqdd.report.generator import compute_mos


WHITELIST = frozenset({"vlm_analyze", "rerun_detector", "dispatch_compression", "accept"})

# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class JudgeClient(ABC):
    @abstractmethod
    def review(self, prompt: str) -> dict[str, Any] | None:
        ...

    def decide(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        """ReAct Agent 决策接口：给定 system + user prompt，返回 JSON 决策。
        默认实现复用 review()（只传 user_prompt）；子类可按需重写。"""
        return self.review(user_prompt)


# ---------------------------------------------------------------------------
# Ollama 实现
# ---------------------------------------------------------------------------


class OllamaJudgeClient(JudgeClient):
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    def review(self, prompt: str) -> dict[str, Any] | None:
        return self._call(JUDGE_SYSTEM_PROMPT, prompt)

    def decide(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        return self._call(system_prompt, user_prompt)

    def _call(self, system: str, user: str) -> dict[str, Any] | None:
        url = f"{self.config.host.rstrip('/')}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
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


# ---------------------------------------------------------------------------
# Mock / Rule-Based 实现（测试 & 降级）
# ---------------------------------------------------------------------------


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

    def decide(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        """Mock 实现：仅在第一步且存在低置信度项时触发一次 vlm_analyze，之后 accept。"""
        try:
            import re

            # 如果历史中已有 vlm_analyze 步骤，则直接 accept，避免重复循环
            if "vlm_analyze" in user_prompt and "observation" in user_prompt:
                return {
                    "thought": "已执行过 VLM 确认，接受当前结果",
                    "action": "accept",
                    "reason": "mock agent：VLM 已确认，结果可信",
                }
            confs = [float(x) for x in re.findall(r'"confidence":\s*([0-9.]+)', user_prompt)]
            low_conf = [c for c in confs if c < 0.7]
            if low_conf:
                return {
                    "thought": "检测到低置信度项，需要 VLM 视觉确认",
                    "action": "vlm_analyze",
                    "degradation_id": None,
                    "reason": f"置信度 {low_conf[0]:.2f} 不足，需要 VLM 确认",
                }
        except Exception:
            pass
        return {
            "thought": "检测结果置信度充足，无需进一步分析",
            "action": "accept",
            "reason": "mock agent：结果可信",
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

    def decide(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        """规则降级：基于简单条件判断是否需要 VLM 或补检。"""
        try:
            import re

            # 检测低置信度项
            confs = [float(x) for x in re.findall(r'"confidence":\s*([0-9.]+)', user_prompt)]
            low_conf_items = [c for c in confs if self.agent_cfg.grey_zone_lower <= c < self.agent_cfg.grey_zone_upper]
            if low_conf_items:
                return {
                    "thought": f"发现 {len(low_conf_items)} 个灰区置信度检测项，规则建议 VLM 确认",
                    "action": "vlm_analyze",
                    "degradation_id": None,
                    "reason": "规则路由：置信度处于灰区",
                }
            # 检测 MOS 偏低但无检出
            m_mos = re.search(r"global_mos[\":\s]+([0-9.]+)", user_prompt)
            m_count = re.search(r"detection_count[\":\s]+([0-9]+)", user_prompt)
            if m_mos and m_count:
                mos = float(m_mos.group(1))
                count = int(m_count.group(1))
                if mos < 4.0 and count == 0:
                    return {
                        "thought": f"MOS={mos:.2f} 偏低但无检出，规则建议补检压缩伪影",
                        "action": "dispatch_compression",
                        "reason": "规则路由：MOS 低但无检出",
                    }
        except Exception:
            pass
        return {
            "thought": "规则审查：结果可信，无需进一步分析",
            "action": "accept",
            "reason": "规则路由：未发现明显矛盾",
        }


# ---------------------------------------------------------------------------
# 解析工具
# ---------------------------------------------------------------------------


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
        degradation_id=raw.get("degradation_id"),
    )


def parse_agent_step(raw: dict[str, Any], step_index: int, latency_ms: float = 0.0) -> AgentStep | None:
    """解析 ReAct Agent 单步输出（Thought + Action）。"""
    action_raw = _parse_action(raw)
    if action_raw is None:
        return None
    return AgentStep(
        step_index=step_index,
        thought=str(raw.get("thought", "")),
        action=action_raw,
        latency_ms=latency_ms,
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


# ---------------------------------------------------------------------------
# 构建 Agent 观察提示词
# ---------------------------------------------------------------------------


def build_agent_observe_prompt(
    degradations: list[DegradationItem],
    report_cfg: Any,
    skipped: list[str],
    step: int,
    max_steps: int,
    history: list[AgentStep],
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt) 供 LLM decide() 使用。"""
    mos, _ = compute_mos(degradations, report_cfg)
    detections = [
        {
            "degradation_id": d.degradation_id,
            "detector": d.detector,
            "confidence": round(d.confidence, 3),
            "degradation_type": d.degradation_type,
            "severity": d.severity,
            "mos_impact": round(d.mos_impact, 3),
        }
        for d in degradations
    ]

    history_lines: list[str] = []
    for s in history:
        history_lines.append(
            f"  步骤{s.step_index}: thought={s.thought!r} "
            f"action={s.action.action} reason={s.action.reason!r} "
            f"observation={s.observation!r}"
        )
    history_str = "\n".join(history_lines) if history_lines else "  （无）"

    system_prompt = AGENT_SYSTEM_PROMPT.format(max_steps=max_steps)
    user_prompt = AGENT_OBSERVE_TEMPLATE.format(
        step=step,
        max_steps=max_steps,
        global_mos=round(mos, 3),
        detection_count=len(degradations),
        detections_json=json.dumps(detections, ensure_ascii=False, indent=2),
        skipped_detectors=json.dumps(skipped, ensure_ascii=False),
        history=history_str,
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# 旧 Judge 接口（保留向后兼容，用于 RuleBasedJudgeClient 降级路径）
# ---------------------------------------------------------------------------


def build_judge_prompt(
    degradations: list[DegradationItem],
    report_cfg: Any,
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
    report_cfg: Any,
    skipped_detectors: list[str] | None = None,
) -> JudgeOutput:
    """旧式 Judge 接口：仅在非 Agent 模式（MockJudgeClient / RuleBasedJudgeClient）时使用。"""
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
