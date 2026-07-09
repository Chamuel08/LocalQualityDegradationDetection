from __future__ import annotations

from lqdd.models.agent import AgentContext
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput


def create_context(
    frame_input: SingleFrameInput,
    scan_output: GlobalScanOutput,
    max_rounds: int = 2,
) -> AgentContext:
    return AgentContext(
        round_index=1,
        max_rounds=max_rounds,
        mode="fast",
        frame_input=frame_input,
        scan_output=scan_output,
    )
