"""业务场景归因（Scenario Attribution）单元测试。"""
from __future__ import annotations

from types import SimpleNamespace

from lqdd.attribution.scenario import attribute_scenarios, scenario_attribution_to_dict


def _deg(degradation_id, degradation_type, detector, severity):
    return SimpleNamespace(
        degradation_id=degradation_id,
        degradation_type=degradation_type,
        detector=detector,
        severity=severity,
        description=f"{degradation_type} detected",
    )


def test_empty_degradations_returns_empty():
    assert attribute_scenarios([]) == []


def test_compression_maps_to_transcode_scenario():
    degs = [_deg("d1", "blockiness", "compression_artifact", "moderate")]
    out = attribute_scenarios(degs)
    assert len(out) == 1
    assert out[0].scenario == "转码"
    assert "blockiness" in out[0].degradation_types
    assert out[0].evidence_refs == ["d1"]
    assert "码率" in out[0].recommendation or "QP" in out[0].recommendation


def test_flicker_maps_to_live_scenario():
    degs = [_deg("d2", "flicker", "temporal_flicker", "critical")]
    out = attribute_scenarios(degs)
    assert out[0].scenario == "直播"
    assert out[0].confidence == 1.0  # critical -> 1.0


def test_face_blur_maps_to_recommendation_cover():
    degs = [_deg("d3", "face_blur", "face_artifact", "moderate")]
    out = attribute_scenarios(degs)
    assert out[0].scenario == "推荐/封面"


def test_edge_bleed_maps_to_composition():
    degs = [_deg("d4", "green_spill", "edge_bleed", "severe")]
    out = attribute_scenarios(degs)
    assert out[0].scenario == "合成/抠图"


def test_hand_anomaly_maps_to_aigc():
    degs = [_deg("d5", "hand_anomaly", "hand_anomaly", "moderate")]
    out = attribute_scenarios(degs)
    assert out[0].scenario == "AIGC"


def test_grouping_by_scenario_and_confidence_order():
    degs = [
        _deg("d1", "blockiness", "compression_artifact", "minor"),
        _deg("d2", "mosaic", "mosaic_artifact", "critical"),  # same scenario 转码, higher severity
        _deg("d3", "face_blur", "face_artifact", "moderate"),  # different scenario
    ]
    out = attribute_scenarios(degs)
    # two distinct scenarios: 转码 (d1+d2), 推荐/封面 (d3)
    scenarios = [a.scenario for a in out]
    assert "转码" in scenarios and "推荐/封面" in scenarios
    # 转码 group aggregates two degradation types
    transcode = next(a for a in out if a.scenario == "转码")
    assert set(transcode.degradation_types) == {"blockiness", "mosaic"}
    assert set(transcode.evidence_refs) == {"d1", "d2"}
    # confidence of 转码 = max(minor=0.4, critical=1.0) = 1.0
    assert transcode.confidence == 1.0
    # sorted by confidence desc: 转码(1.0) before 推荐/封面(0.7)
    assert out[0].confidence >= out[1].confidence


def test_unknown_degradation_skipped():
    degs = [_deg("d9", "unknown_weird_type", "mystery_detector", "moderate")]
    out = attribute_scenarios(degs)
    assert out == []


def test_to_dict_serialization():
    degs = [_deg("d1", "blockiness", "compression_artifact", "moderate")]
    out = attribute_scenarios(degs)
    d = scenario_attribution_to_dict(out[0])
    assert set(d.keys()) == {"scenario", "confidence", "degradation_types", "evidence_refs", "recommendation"}
    assert d["scenario"] == "转码"
