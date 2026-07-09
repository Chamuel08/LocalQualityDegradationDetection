from lqdd.models.agent import VLMResult
from lqdd.models.report import DegradationItem, Evidence, RootCauseHypothesis
from lqdd.vlm.fuse import fuse_degradation


def _deg(conf: float = 0.55) -> DegradationItem:
    return DegradationItem(
        degradation_id="d1",
        region_type="edge",
        degradation_type="green_spill",
        severity="minor",
        confidence=conf,
        mos_impact=-0.2,
        bbox=[0, 0, 10, 10],
        frame_indices=[0],
        description="test",
        detector="edge_bleed",
        evidence=Evidence("m", "metric", 0.1, 0.05, "测试"),
        root_cause_hypothesis=RootCauseHypothesis("matting_error", 0.7),
    )


def test_fuse_agree() -> None:
    vlm = VLMResult(
        region_type="edge",
        degradation_type="green_spill",
        is_degraded=True,
        vlm_confidence=0.8,
        vlm_severity="moderate",
        reasoning="可见绿边",
        mos_impact_estimate=-0.3,
        root_cause="matting_error",
        ux_impact="不自然",
    )
    fused = fuse_degradation(_deg(), vlm)
    assert fused.vlm_reasoning["fusion_decision"] == "agree"
    assert fused.confidence > 0.55


def test_fuse_vlm_override() -> None:
    deg = _deg(0.5)
    deg.severity = "good"
    vlm = VLMResult(
        region_type="edge",
        degradation_type="green_spill",
        is_degraded=True,
        vlm_confidence=0.9,
        vlm_severity="severe",
        reasoning="明显绿边",
        mos_impact_estimate=-0.4,
        root_cause="matting_error",
        ux_impact="严重影响",
    )
    fused = fuse_degradation(deg, vlm)
    assert fused.vlm_reasoning["fusion_decision"] == "vlm_override"
