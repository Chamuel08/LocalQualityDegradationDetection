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
    mild_blockiness_threshold: float = 1.02
    texture_var_reference: float = 2400.0
    texture_loss_threshold: float = 0.38
    edge_block_ratio_threshold: float = 1.012


@dataclass
class BlurConfig:
    texture_var_reference: float = 2400.0
    texture_loss_threshold: float = 0.42
    min_foreground_ratio: float = 0.05


@dataclass
class MosaicConfig:
    score_threshold: float = 0.48
    localize_threshold: float = 0.35


@dataclass
class BandingConfig:
    score_threshold: float = 0.38
    localize_threshold: float = 0.30


@dataclass
class BackgroundArtifactConfig:
    blockiness_threshold: float = 1.35
    color_cast_threshold: float = 0.18
    min_background_ratio: float = 0.12


@dataclass
class HairTextureConfig:
    relative_laplacian_threshold: float = 0.55
    max_hair_laplacian_var: float = 35.0
    min_hair_pixels: int = 80


@dataclass
class FaceArtifactConfig:
    overexposure_ratio_threshold: float = 0.25
    blur_laplacian_threshold: float = 85.0
    min_face_pixels: int = 300


@dataclass
class HandAnomalyConfig:
    edge_density_threshold: float = 0.14
    finger_spread_threshold: float = 2.8
    min_hand_pixels: int = 250


@dataclass
class ReportConfig:
    base_mos: float = 4.5
    decay_factor: float = 0.7
    system_version: str = "0.1.0"
    # MOS 拟合模型选择：
    #   "rule"      - 默认，使用硬编码 mos_impact + 衰减公式（当前实现）
    #   "clip_iqa"  - 接入开源 CLIP-IQA 模型（需安装 clip-iqa 依赖）
    #   "internal"  - 接入内部拟合模型（需实现 lqdd.mos.internal_model）
    # 扩展方式：在 compute_mos() 中按此字段分发到对应实现。
    mos_model: str = "rule"


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
    blur: BlurConfig
    mosaic: MosaicConfig
    banding: BandingConfig
    background_artifact: BackgroundArtifactConfig
    hair_texture: HairTextureConfig
    face_artifact: FaceArtifactConfig
    hand_anomaly: HandAnomalyConfig
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
        blur=BlurConfig(),
        mosaic=MosaicConfig(),
        banding=BandingConfig(),
        background_artifact=BackgroundArtifactConfig(),
        hair_texture=HairTextureConfig(),
        face_artifact=FaceArtifactConfig(),
        hand_anomaly=HandAnomalyConfig(),
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
        blur=_section(data.get("blur"), BlurConfig, defaults.blur),
        mosaic=_section(data.get("mosaic"), MosaicConfig, defaults.mosaic),
        banding=_section(data.get("banding"), BandingConfig, defaults.banding),
        background_artifact=_section(
            data.get("background_artifact"), BackgroundArtifactConfig, defaults.background_artifact
        ),
        hair_texture=_section(data.get("hair_texture"), HairTextureConfig, defaults.hair_texture),
        face_artifact=_section(data.get("face_artifact"), FaceArtifactConfig, defaults.face_artifact),
        hand_anomaly=_section(data.get("hand_anomaly"), HandAnomalyConfig, defaults.hand_anomaly),
        report=_section(data.get("report"), ReportConfig, defaults.report),
        agent=_section(data.get("agent"), AgentConfig, defaults.agent),
        vlm=_section(data.get("vlm"), VLMConfig, defaults.vlm),
        judge=_section(data.get("judge"), JudgeConfig, defaults.judge),
    )
    return _apply_env(config)
