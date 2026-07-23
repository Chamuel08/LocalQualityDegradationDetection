"""业务场景归因（Scenario Attribution）。

把检测到的劣化映射到业务场景与修复建议，让检测结果可执行于下游业务。
"""
from lqdd.attribution.scenario import (
    ScenarioAttribution,
    attribute_scenarios,
    scenario_attribution_to_dict,
)

__all__ = [
    "ScenarioAttribution",
    "attribute_scenarios",
    "scenario_attribution_to_dict",
]
