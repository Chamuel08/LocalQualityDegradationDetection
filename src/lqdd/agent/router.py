from __future__ import annotations

from lqdd.config.loader import AgentConfig
from lqdd.models.agent import AgentContext, RoutingDecision
from lqdd.models.enums import RegionType
from lqdd.models.inputs import GlobalScanOutput

DETECTOR_MAP = {
    int(RegionType.EDGE): "edge_bleed",
    int(RegionType.BACKGROUND): "compression_artifact",
}


def _confidence_band(score: float, cfg: AgentConfig) -> str:
    if score >= cfg.high_confidence_threshold:
        return "high"
    if score >= cfg.grey_zone_lower:
        return "grey"
    return "low"


def route_nominations(scan: GlobalScanOutput, cfg: AgentConfig, ctx: AgentContext) -> list[str]:
    """Return list of detector names to dispatch."""
    dispatched: list[str] = []
    decisions: list[RoutingDecision] = []

    for nom in scan.nominations:
        detector = None
        for d in nom.suggested_detectors:
            if d in ("edge_bleed", "compression_artifact"):
                detector = d
                break
        if not detector:
            detector = DETECTOR_MAP.get(nom.region_type)

        band = _confidence_band(nom.confidence, cfg)
        if band == "low":
            decisions.append(
                RoutingDecision(
                    target_detector=detector,
                    decision="skip",
                    reason=f"置信度 {nom.confidence:.2f} 低于灰区下界",
                    confidence_band="low",
                    region_type=str(nom.region_type),
                    anomaly_score=nom.anomaly_score,
                )
            )
            continue

        decision_type = "vlm_pending" if band == "grey" else "dispatch"
        decisions.append(
            RoutingDecision(
                target_detector=detector,
                decision=decision_type,
                reason=f"提名 anomaly={nom.anomaly_score:.2f} band={band}",
                confidence_band=band,
                region_type=str(nom.region_type),
                anomaly_score=nom.anomaly_score,
            )
        )
        if detector and detector not in dispatched:
            if len(dispatched) < cfg.max_detectors_per_frame:
                dispatched.append(detector)

    if not dispatched:
        dispatched = ["edge_bleed", "compression_artifact"]

    ctx.routing_decisions = decisions
    ctx.dispatched_detectors = dispatched
    return dispatched


def is_grey_confidence(confidence: float, cfg: AgentConfig) -> bool:
    return cfg.grey_zone_lower <= confidence < cfg.grey_zone_upper
