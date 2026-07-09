from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GlobalScanConfig:
    edge_expand_px: int = 10
    nomination_threshold: float = 0.3
    text_ui_ratio_critical: float = 0.6
    subtitle_band_ratio: float = 0.15
    text_edge_density: float = 0.12


@dataclass
class EdgeBleedConfig:
    green_spill_minor: float = 0.05
    green_spill_moderate: float = 0.15
    green_spill_critical: float = 0.30
    green_channel_threshold: float = 0.15
    delta_e_spill_threshold: float = 10.0
    spill_ratio_minor: float = 0.05


@dataclass
class CompressionConfig:
    blockiness_threshold: float = 1.8
    blockiness_coarse_threshold: float = 1.4


@dataclass
class ReportConfig:
    base_mos: float = 4.5
    decay_factor: float = 0.7
    system_version: str = "0.1.0"


@dataclass
class AgentConfig:
    enabled: bool = True
    max_rounds: int = 2
    high_confidence_threshold: float = 0.7
    grey_zone_lower: float = 0.4
    grey_zone_upper: float = 0.7
    max_detectors_per_frame: int = 5
    hard_decision_threshold: float = 0.55


@dataclass
class VLMConfig:
    provider: str = "ollama"
    model: str = "qwen2.5-vl:7b"
    host: str = "http://localhost:11434"
    timeout_ms: int = 2000
    max_calls_per_frame: int = 3


@dataclass
class JudgeConfig:
    provider: str = "ollama"
    model: str = "qwen2.5:1.5b"
    host: str = "http://localhost:11434"
    timeout_ms: int = 1500


@dataclass
class AppConfig:
    global_scan: GlobalScanConfig
    edge_bleed: EdgeBleedConfig
    compression: CompressionConfig
    report: ReportConfig
    agent: AgentConfig
    vlm: VLMConfig
    judge: JudgeConfig


def _section(data: dict[str, Any], cls: type, defaults: Any) -> Any:
    raw = data or {}
    fields = {k: raw.get(k, getattr(defaults, k)) for k in defaults.__dataclass_fields__}
    return cls(**fields)


def _apply_env(config: AppConfig) -> AppConfig:
    if os.environ.get("LQDD_AGENT_ENABLED", "").lower() in ("0", "false", "no"):
        config.agent.enabled = False
    if host := os.environ.get("OLLAMA_HOST"):
        config.vlm.host = host
        config.judge.host = host
    if model := os.environ.get("LQDD_VLM_MODEL"):
        config.vlm.model = model
    if model := os.environ.get("LQDD_JUDGE_MODEL"):
        config.judge.model = model
    if config.agent.enabled:
        config.report.system_version = "1.0.0"
    return config


def load_config(path: Path | None = None) -> AppConfig:
    defaults = AppConfig(
        global_scan=GlobalScanConfig(),
        edge_bleed=EdgeBleedConfig(),
        compression=CompressionConfig(),
        report=ReportConfig(),
        agent=AgentConfig(),
        vlm=VLMConfig(),
        judge=JudgeConfig(),
    )
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend([Path("config.yaml"), Path("config.example.yaml")])

    data: dict[str, Any] = {}
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            break

    config = AppConfig(
        global_scan=_section(data.get("global_scan"), GlobalScanConfig, defaults.global_scan),
        edge_bleed=_section(data.get("edge_bleed"), EdgeBleedConfig, defaults.edge_bleed),
        compression=_section(data.get("compression"), CompressionConfig, defaults.compression),
        report=_section(data.get("report"), ReportConfig, defaults.report),
        agent=_section(data.get("agent"), AgentConfig, defaults.agent),
        vlm=_section(data.get("vlm"), VLMConfig, defaults.vlm),
        judge=_section(data.get("judge"), JudgeConfig, defaults.judge),
    )
    return _apply_env(config)
