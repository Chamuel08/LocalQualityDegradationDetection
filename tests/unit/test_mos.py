import pytest

from lqdd.config.loader import ReportConfig
from lqdd.models.report import DegradationItem, Evidence, RootCauseHypothesis
from lqdd.report.generator import compute_mos


def _item(deg_id: str, impact: float) -> DegradationItem:
    return DegradationItem(
        degradation_id=deg_id,
        region_type="edge",
        degradation_type="green_spill",
        severity="minor",
        confidence=0.8,
        mos_impact=impact,
        bbox=[0, 0, 10, 10],
        frame_indices=[0],
        description="test",
        detector="edge_bleed",
        evidence=Evidence(
            method="test",
            metric="spill_ratio",
            value=0.1,
            threshold=0.05,
            detail="测试",
        ),
        root_cause_hypothesis=RootCauseHypothesis(cause="matting_error", confidence=0.7),
    )


def test_mos_no_degradations() -> None:
    mos, breakdown = compute_mos([], ReportConfig())
    assert mos == 4.5
    assert breakdown.total_penalty == 0.0


def test_mos_single_penalty() -> None:
    mos, breakdown = compute_mos([_item("a", -0.2)], ReportConfig())
    assert mos == pytest.approx(4.3, abs=0.01)
    assert breakdown.total_penalty == pytest.approx(-0.2, abs=0.01)


def test_mos_decay_second_item() -> None:
    items = [_item("a", -0.4), _item("b", -0.2)]
    mos, breakdown = compute_mos(items, ReportConfig())
    expected_total = -0.4 + (-0.2 * 0.7)
    assert breakdown.total_penalty == pytest.approx(expected_total, abs=0.01)
    assert mos == pytest.approx(4.5 + expected_total, abs=0.01)
