from __future__ import annotations

from copy import deepcopy

from lqdd.models.agent import VLMResult
from lqdd.models.report import DegradationItem


def fuse_degradation(deg: DegradationItem, vlm: VLMResult) -> DegradationItem:
    out = deepcopy(deg)
    det_conf = deg.confidence
    vlm_conf = vlm.vlm_confidence
    agree = (deg.severity != "good" and vlm.is_degraded) or (deg.severity == "good" and not vlm.is_degraded)

    if agree:
        final_conf = 0.4 * det_conf + 0.6 * vlm_conf
        fusion = "agree"
    elif vlm_conf > 0.8:
        final_conf = vlm_conf
        fusion = "vlm_override"
        if vlm.is_degraded:
            out.severity = vlm.vlm_severity
            out.mos_impact = min(deg.mos_impact, vlm.mos_impact_estimate)
    elif det_conf > 0.8:
        final_conf = det_conf
        fusion = "detector_override"
    else:
        final_conf = min(det_conf, vlm_conf)
        fusion = "uncertain"
        out.mos_impact = min(deg.mos_impact, vlm.mos_impact_estimate)

    out.confidence = round(final_conf, 4)
    out.vlm_reasoning = {
        "reasoning": vlm.reasoning,
        "vlm_confidence": vlm_conf,
        "ux_impact": vlm.ux_impact,
        "fusion_decision": fusion,
    }
    return out


def fuse_all(
    degradations: list[DegradationItem],
    vlm_results: list[VLMResult],
) -> list[DegradationItem]:
    if not vlm_results:
        return list(degradations)

    by_type: dict[str, VLMResult] = {}
    for v in vlm_results:
        by_type[v.degradation_type] = v

    merged: list[DegradationItem] = []
    for deg in degradations:
        vlm = by_type.get(deg.degradation_type)
        if vlm and deg.vlm_reasoning is None:
            merged.append(fuse_degradation(deg, vlm))
        else:
            merged.append(deg)
    return merged
