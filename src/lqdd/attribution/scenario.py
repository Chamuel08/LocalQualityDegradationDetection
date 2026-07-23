"""业务场景归因（Scenario Attribution）。

把检测到的劣化映射到业务场景（点播 / 直播 / 推荐 / 转码 / 采集 / 合成 / AIGC），
并给出可落地的修复建议。纯规则映射，无外部依赖，不参与 MOS 计算。

让检测结果从「检出劣化」升级到「可执行的业务建议」，打通检测与下游业务。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lqdd.models.report import DegradationItem


@dataclass
class ScenarioAttribution:
    """单条业务场景归因：一类劣化 → 一个业务场景 → 一条修复建议。"""

    scenario: str                       # 业务场景：点播/直播/推荐/转码/采集/合成/AIGC
    confidence: float                  # 归因置信度 [0, 1]（按命中劣化的严重度/数量估计）
    degradation_types: list[str]       # 命中的劣化类型
    evidence_refs: list[str]           # 关联的 degradation_id
    recommendation: str                # 可落地的修复建议


# ---------------------------------------------------------------------------
# 规则表：劣化类型 / 检测器 → (业务场景, 修复建议)
# 优先按 degradation_type 匹配，再回退到 detector。
# ---------------------------------------------------------------------------

_DEGRADATION_TYPE_RULES: dict[str, tuple[str, str]] = {
    "blockiness": (
        "转码",
        "压缩块效应明显；建议提升目标码率或降低 QP，必要时启用 ROI 编码集中码率到主体",
    ),
    "blur": (
        "采集/拍摄",
        "主体区域模糊；建议采集端对焦校验，或后处理超分/锐化增强",
    ),
    "mosaic": (
        "转码",
        "过度下采样或低清源；建议提升输出分辨率，或接入升清模型",
    ),
    "banding": (
        "转码",
        "色深不足导致色带；建议 10bit 编码或提升码率，避免过度量化",
    ),
    "green_spill": (
        "合成/抠图",
        "边缘绿幕溢色；建议复查 matting 边缘与溢色抑制后处理",
    ),
    "hair_texture_loss": (
        "转码",
        "高频细节丢失；建议提升码率或启用高频保留工具（如 B 帧自适应量化）",
    ),
    "overexposure": (
        "采集/拍摄",
        "面部过曝；建议采集端曝光控制或后处理色调映射",
    ),
    "face_blur": (
        "推荐/封面",
        "面部画质差影响封面点击率；建议换帧或对面部做超分增强",
    ),
    "background_artifact": (
        "转码",
        "背景编码劣化；建议 ROI 编码，把码率集中到主体区域",
    ),
    "hand_anomaly": (
        "AIGC",
        "AI 生成几何异常（如多指）；建议生成模型复查或后处理修复",
    ),
    "flicker": (
        "直播",
        "时域闪烁；建议检查编码器 VBR/CBR 稳态参数与帧间量化一致性",
    ),
}

_DETECTOR_RULES: dict[str, tuple[str, str]] = {
    "compression_artifact": (
        "转码",
        "压缩伪影/纹理损失；建议提升码率或降低 QP，必要时启用 ROI 编码",
    ),
    "blur_artifact": (
        "采集/拍摄",
        "主体模糊；建议采集端对焦校验或后处理超分增强",
    ),
    "mosaic_artifact": (
        "转码",
        "马赛克/像素化；建议提升输出分辨率或接入升清模型",
    ),
    "banding_artifact": (
        "转码",
        "色带伪影；建议 10bit 编码或提升码率",
    ),
    "edge_bleed": (
        "合成/抠图",
        "边缘溢色；建议复查 matting/合成边缘处理",
    ),
    "hair_texture": (
        "转码",
        "发丝细节损失；建议提升码率或启用高频保留",
    ),
    "face_artifact": (
        "推荐/封面",
        "面部伪影影响封面质量；建议换帧或面部超分",
    ),
    "background_artifact": (
        "转码",
        "背景块效应/色彩漂移；建议 ROI 编码集中码率到主体",
    ),
    "hand_anomaly": (
        "AIGC",
        "手部几何异常；建议生成模型复查或后处理修复",
    ),
}

# 严重度 → 权重（用于归因置信度估计）
_SEVERITY_WEIGHT = {"minor": 0.4, "moderate": 0.7, "severe": 0.85, "critical": 1.0}


def _rule_for(deg: DegradationItem) -> tuple[str, str] | None:
    """按 degradation_type 优先、detector 回退，查归因规则。"""
    if deg.degradation_type in _DEGRADATION_TYPE_RULES:
        return _DEGRADATION_TYPE_RULES[deg.degradation_type]
    if deg.detector in _DETECTOR_RULES:
        return _DETECTOR_RULES[deg.detector]
    return None


def attribute_scenarios(degradations: list[DegradationItem]) -> list[ScenarioAttribution]:
    """对一帧/一 clip 的劣化列表做业务场景归因。

    策略：按业务场景聚合，每个场景产出一条 ScenarioAttribution，
    置信度取该场景下命中劣化的最大严重度权重（无命中劣化时为 0）。

    Returns:
        按 confidence 降序排列的归因列表；无劣化时返回 []。
    """
    if not degradations:
        return []

    # scenario -> (degradation_types, evidence_refs, max_weight, recommendation)
    buckets: dict[str, dict[str, Any]] = {}
    for deg in degradations:
        rule = _rule_for(deg)
        if rule is None:
            continue
        scenario, recommendation = rule
        w = _SEVERITY_WEIGHT.get(deg.severity, 0.5)
        b = buckets.setdefault(
            scenario,
            {"types": [], "refs": [], "max_w": 0.0, "rec": recommendation},
        )
        if deg.degradation_type and deg.degradation_type not in b["types"]:
            b["types"].append(deg.degradation_type)
        b["refs"].append(deg.degradation_id)
        b["max_w"] = max(b["max_w"], w)

    results = [
        ScenarioAttribution(
            scenario=scenario,
            confidence=round(b["max_w"], 3),
            degradation_types=b["types"],
            evidence_refs=b["refs"],
            recommendation=b["rec"],
        )
        for scenario, b in buckets.items()
    ]
    results.sort(key=lambda x: x.confidence, reverse=True)
    return results


def scenario_attribution_to_dict(a: ScenarioAttribution) -> dict[str, Any]:
    return {
        "scenario": a.scenario,
        "confidence": a.confidence,
        "degradation_types": a.degradation_types,
        "evidence_refs": a.evidence_refs,
        "recommendation": a.recommendation,
    }
