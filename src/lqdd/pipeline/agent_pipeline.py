from __future__ import annotations

from typing import Any

from lqdd.agent.orchestrator import AgentOrchestrator
from lqdd.config.loader import AppConfig
from lqdd.models.inputs import SingleFrameInput
from lqdd.models.report import QualityReport
from lqdd.vlm.client import VLMClient


class AgentPipeline:
    def __init__(
        self,
        config: AppConfig,
        vlm_client: VLMClient | None = None,
        judge_client: Any = None,
    ) -> None:
        self.orchestrator = AgentOrchestrator(config, vlm_client=vlm_client, judge_client=judge_client)

    def run(self, frame_input: SingleFrameInput) -> QualityReport:
        return self.orchestrator.run(frame_input)
